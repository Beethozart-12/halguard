"""HalGuard 命令行入口。

用法：
  halguard serve                 # 启动代理服务（默认 0.0.0.0:8849）
  halguard ingest [PATH]         # 摄入知识库（默认 ./knowledge），构建检索索引
  halguard verify --text "..."   # 一次性验证一段文本的幻觉风险
  halguard status                # 显示配置与索引状态
"""
from __future__ import annotations

import argparse
import sys
import os

from .config import HalGuardConfig
from .retrieval import RetrievalIndex
from . import verify as V


def _load_or_new_retrieval(cfg: HalGuardConfig) -> RetrievalIndex:
    try:
        return RetrievalIndex.load(cfg.index_path, cfg.embedder, cfg.st_model)
    except FileNotFoundError:
        return RetrievalIndex.new(cfg.embedder, cfg.st_model)


def cmd_serve(args):
    import uvicorn
    cfg = HalGuardConfig.load()
    retrieval = _load_or_new_retrieval(cfg)
    from .server import build_app
    app = build_app(cfg, retrieval)
    print(f"[HalGuard] 监听 http://{cfg.listen_host}:{cfg.listen_port}")
    print(f"[HalGuard] 后端 LLM: {cfg.backend_base_url}  model={cfg.backend_model}")
    print(f"[HalGuard] 知识库块数: {retrieval.count()}  嵌入器: {retrieval.embedder.name}")
    print(f"[HalGuard] 把客户端的 OpenAI base_url 指向 http://localhost:{cfg.listen_port}/v1 即可启用抗幻觉。")
    uvicorn.run(app, host=cfg.listen_host, port=cfg.listen_port, log_level="info")


def cmd_ingest(args):
    cfg = HalGuardConfig.load()
    path = args.path or cfg.kb_path
    retrieval = _load_or_new_retrieval(cfg)
    n = retrieval.ingest(path)
    retrieval.save(cfg.index_path)
    print(f"[HalGuard] 已摄入 {n} 个文本块 -> {cfg.index_path}")
    print(f"[HalGuard] 嵌入器: {retrieval.embedder.name}")


def cmd_verify(args):
    cfg = HalGuardConfig.load()
    text = args.text or sys.stdin.read()
    if not text.strip():
        print("错误：请提供 --text 或管道输入。", file=sys.stderr)
        sys.exit(1)

    chunks = []
    chunk_embs = None
    try:
        retrieval = RetrievalIndex.load(cfg.index_path, cfg.embedder, cfg.st_model)
        if retrieval.count() > 0:
            chunks, chunk_embs = retrieval.query(text, top_k=cfg.top_k)
    except FileNotFoundError:
        pass

    if not chunks and args.context:
        chunks = [args.context]
        # 用相同嵌入器编码上下文
        retr = _load_or_new_retrieval(cfg)
        chunk_embs = retr.embedder.embed([args.context])

    claims = V.extract_claims(text)
    if not chunks:
        print("警告：未找到知识库索引，也无 --context，无法对照验证（结果仅供参考）。")
        report = [{"claim": c, "support": 0.0, "unsupported": True} for c in claims]
    else:
        retr = _load_or_new_retrieval(cfg)
        report = V.score_claims(claims, chunk_embs, retr.embedder, cfg.support_threshold)
    risk = V.compute_risk(report)
    print(f"断言数: {len(claims)}  整体风险: {risk:.2f}")
    for r in report:
        tag = "❌ 疑似幻觉" if r["unsupported"] else "✅ 已支持"
        print(f"  [{tag}] (支持度 {r['support']:.2f}) {r['claim']}")
    if risk > cfg.risk_threshold:
        print("\n" + V.build_warning(report, risk))


def cmd_status(args):
    cfg = HalGuardConfig.load()
    try:
        retrieval = RetrievalIndex.load(cfg.index_path, cfg.embedder, cfg.st_model)
        kb = retrieval.count()
        emb = retrieval.embedder.name
    except FileNotFoundError:
        kb, emb = 0, "(未构建索引)"
    print("=== HalGuard 配置 ===")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  知识库块数: {kb}")
    print(f"  嵌入器: {emb}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="halguard", description="本地 LLM 抗幻觉中间件")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="启动代理服务")
    s.set_defaults(func=cmd_serve)

    i = sub.add_parser("ingest", help="摄入知识库")
    i.add_argument("path", nargs="?", default=None, help="知识库目录或文件")
    i.set_defaults(func=cmd_ingest)

    v = sub.add_parser("verify", help="一次性验证文本")
    v.add_argument("--text", default=None, help="待验证文本（或用管道）")
    v.add_argument("--context", default=None, help="对照上下文（无知识库索引时使用）")
    v.set_defaults(func=cmd_verify)

    st = sub.add_parser("status", help="显示状态")
    st.set_defaults(func=cmd_status)
    return p


def main(argv=None):
    # Windows 默认 GBK 控制台无法输出 emoji/中文，强制 UTF-8 以避免崩溃
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

"""从 hf-mirror 直接拉取 sentence-transformers 模型到本地（绕开中国大陆网络限制）。

背景：huggingface.co 直连被墙；`snapshot_download(endpoint=hf-mirror)` 会产出空 snapshot；
ModelScope 被沙箱拦截；清华镜像 URL 结构与 transformers 不兼容。本脚本用原生 urllib 逐文件
下载，跟随重定向、长超时、自动重试，并跳过冗余的 PyTorch/TF 权重（已有 safetensors）。

用法：
    python download_model.py
    HF_ENDPOINT=https://hf-mirror.com python download_model.py   # 若需换镜像
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request

BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 默认落到仓库内 halguard_models/ 下，便于与 HalGuard 索引配套使用
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "halguard_models", "paraphrase-multilingual-MiniLM-L12-v2")
# sentence-transformers 用 model.safetensors 即可，跳过 PyTorch/TF 冗余权重
SKIP = {"pytorch_model.bin", "tf_model.h5"}


def get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=timeout)  # 默认跟随 3xx 重定向


def download(url: str, dst: str, timeout: int = 1800, retries: int = 2) -> bool:
    last = None
    for attempt in range(retries + 1):
        try:
            with get(url, timeout=timeout) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  retry {attempt + 1}/{retries} on {os.path.basename(dst)}: {e}")
    print(f"  FAILED {os.path.basename(dst)}: {last}")
    return False


def main() -> None:
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    with get(f"{BASE}/api/models/{REPO}/tree/main") as r:
        items = json.loads(r.read())
    paths = [it["path"] for it in items
             if it.get("type") == "file" and it["path"] not in SKIP]
    print(f"FILES({len(paths)}): {paths}")

    for p in paths:
        dst = os.path.join(OUT, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"{BASE}/{REPO}/resolve/main/{p}"
        if download(url, dst):
            print(f"OK  {p}  ({os.path.getsize(dst)} bytes)")
        else:
            print(f"FAIL {p}")

    print("DONE ->", OUT)


if __name__ == "__main__":
    main()

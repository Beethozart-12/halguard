# HalGuard — 本地 LLM/SLM 抗幻觉中间件

> 当你的本地 AI（Ollama / LM Studio / vLLM / llama.cpp，均暴露 OpenAI 兼容 API）在运行时，HalGuard 也随之运行：所有请求先经过它，通过 **检索增强接地（RAG）** 与 **生成后验证** 两条防线，降低幻觉率。

---

## 它如何降低幻觉？

本地小模型（SLM）幻觉主要源于「缺乏依据地编造事实」。HalGuard 在模型生成前/后各布一道防线：

1. **检索增强接地（RAG Grounding）** — 生成前，从你提供的本地知识库检索相关事实，注入 system 上下文，并强制模型「仅依据上下文作答、并对事实标注来源」。模型有依据可查，就少编造。
2. **生成后验证（Post-generation Verification）** — 生成后，抽取回答中的事实性断言，逐条与检索到的上下文计算 **语义支持度**（余弦相似度）。支持度低于阈值的断言被标记为「疑似幻觉」；整体风险 = 不受支持断言占比。
3. **处置策略** — 按风险阈值执行：
   - `flag`（默认）：在回复末尾追加警告，列出疑似幻觉断言及支持度。
   - `redact`：直接移除/替换不受支持的断言。
   - `reask`：把不受支持的断言清单回灌给模型重新作答（仅用上下文）。
   - `none`：只记录，不改动回复。
4. **可观测** — 内置轻量看板 `/dashboard`，记录每次回答的风险评分与触发动作。

```
应用/客户端
   │  OpenAI 兼容请求 (base_url 指向 HalGuard)
   ▼
┌──────────── HalGuard (常驻) ────────────┐
│ 1. 检索知识库 → 上下文                    │
│ 2. 注入 system：「仅依上下文作答」         │
│ 3. 转发到本地 LLM 生成                    │
│ 4. 抽取断言 → 语义支持度打分 → 标记幻觉    │
│ 5. 按策略 flag/redact/reask               │
└────────────────────┬────────────────────┘
                     │
                     ▼
              本地 LLM (Ollama/LM Studio/...)
```

---

## 安装

```bash
cd halguard
pip install -e .
# 可选：更高质量的语义嵌入（推荐用于生产）
pip install -r requirements-optional.txt
```

> 不使用 sentence-transformers 时，自动回退到 **TF-IDF 嵌入 + 纯 numpy 向量库**，零重依赖即可运行（适合轻量/测试）。

## 快速开始

```bash
# 1) 把你的资料放进 knowledge/（.txt / .md / .json），或指定目录
# 2) 构建检索索引
halguard ingest

# 3) 启动代理（默认 0.0.0.0:8849）
halguard serve
```

然后把任何 OpenAI 客户端 / 应用的 `base_url` 指向 `http://localhost:8849/v1`，`api_key` 随意填（如 `ollama`），`model` 填你的本地模型名。本地 LLM 运行的同时，HalGuard 就在工作。

一次性验证某段文本（无需启动服务）：

```bash
halguard verify --text "巴黎是德国的首都。" --context "巴黎是法国的首都。"
# 断言数: 1  整体风险: 1.00
#   [❌ 疑似幻觉] (支持度 0.00) 巴黎是德国的首都。
```

查看状态：`halguard status`

## 让 HalGuard「随 AI 一起运行」

HalGuard 是常驻代理，最可靠的做法是与本地 LLM 一起启动。任选其一：

- **Linux/macOS**：用 `systemd` / `launchd` / `supervisor` 同时拉起 `ollama serve`（或你的推理服务）与 `halguard serve`。
- **Windows**：把两个程序放进同一个开机/启动脚本，或用 `nssm` 注册为服务。
- **Docker**：在 `docker-compose.yml` 中同时编排推理服务与 `halguard serve`，让应用只连 HalGuard 端口。

只要客户端指向 HalGuard 的端口，就保证「本地 AI 运行时，抗幻觉同步生效」。

## 配置

所有项可用环境变量 `HALGUARD_*` 覆盖（见 `halguard/config.py`）：

| 配置 | 默认 | 说明 |
|------|------|------|
| `backend_base_url` | `http://localhost:11434/v1` | 本地 LLM 的 OpenAI 兼容地址 |
| `backend_model` | `llama3.1:8b` | 实际使用的本地模型名 |
| `listen_port` | `8849` | HalGuard 代理端口 |
| `embedder` | `auto` | `auto`/`st`/`tfidf` |
| `top_k` | `5` | 每次检索的上下文块数 |
| `support_threshold` | `0.35` | 单条断言支持度下限（低于即疑似幻觉）|
| `risk_threshold` | `0.30` | 整体风险触发动作的下限 |
| `action_on_risk` | `flag` | `flag`/`redact`/`reask`/`none` |
| `append_warning` | `true` | flag 时是否在回复末尾追加警告 |

## 局限与说明

- 验证依赖「检索到的上下文」作为事实基准。**知识库越准、越全，抗幻觉效果越好**。
- TF-IDF 嵌入是关键词级，语义泛化弱；生产环境务必安装 `sentence-transformers` 获得句级语义嵌入。
- 本工具是**降低**而非**消除**幻觉：它把「无依据的编造」暴露在风险评分中，并尽量约束模型「有依据才说」。

## License

MIT

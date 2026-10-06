# HalGuard — 本地 LLM/SLM 抗幻觉中间件

> 当你的本地 AI（Ollama / LM Studio / vLLM / llama.cpp，均暴露 OpenAI 兼容 API）在运行时，HalGuard 也随之运行：所有请求先经过它，通过 **检索增强接地（RAG）** 与 **生成后验证** 两条防线，降低幻觉率。

---

## 它如何降低幻觉？

本地小模型（SLM）幻觉主要源于「缺乏依据地编造事实」。HalGuard 在模型生成前/后各布一道防线：

1. **检索增强接地（RAG Grounding）** — 生成前，从你提供的本地知识库检索相关事实，注入 system 上下文，并强制模型「仅依据上下文作答、并对事实标注来源」。模型有依据可查，就少编造。
2. **生成后验证（Post-generation Verification）** — 生成后，抽取回答中的事实性断言，逐条与检索到的上下文计算 **语义支持度**（余弦相似度 + 灰度区字符 bigram 重合度的混合判定，专门捕捉"同主谓、换实体"的篡改句）。支持度低于阈值的断言被标记为「疑似幻觉」；整体风险 = 不受支持断言占比。
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

> 上面的示例输出以**干净环境**（未安装 sentence-transformers、未构建索引）为准：
> 安装 st 后支持度数值会不同（语义嵌入对中文也有相似度），已构建索引时输出以索引命中块为准。
> 另外 `--context` 仅在「无索引」或「索引查询为空」时作为对照；索引存在时会被忽略——
> 若想强制对照指定文本，请把该文本放入临时目录后 `halguard ingest <目录>`。

查看状态：`halguard status`

## 一键部署（Docker Compose）

前置：装好 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，并把嵌入模型放到 `./halguard_models/paraphrase-multilingual-MiniLM-L12-v2/`（仓库 `download_model.py` 可下载）。

```bash
# 一键启动（首次会自动构建镜像并摄入 knowledge/ 建索引）
docker compose up -d

# 查看日志
docker compose logs -f halguard
```

默认配置：

- HalGuard 代理监听 `http://localhost:8849/v1`，把客户端 `base_url` 指过来即可。
- 后端 LLM 默认指向**宿主机上的 Ollama**（`host.docker.internal:11434`）。
- 嵌入模型从挂载的 `/models` 本地加载，**不联网下载**（国内友好）。
- 检索索引持久化在 named volume `halguard_index`，重建容器不丢。

变体用法：

```bash
# 连 Ollama 容器一起一键起（拉起 halguard + ollama 两个服务）
docker compose --profile ollama up -d
# 然后让 HalGuard 指向容器内的 Ollama 并拉一次模型
HALGUARD_BACKEND_BASE_URL=http://ollama:11434/v1 docker compose --profile ollama up -d
docker compose --profile ollama exec ollama ollama pull llama3.1:8b

# 更换后端模型 / 配置，全部通过环境变量覆盖（见下方「配置」）
HALGUARD_BACKEND_MODEL=qwen2.5:7b docker compose up -d
```

> 更新知识库后执行 `docker compose restart halguard`——若索引缺失会自动重新摄入；若想强制重建，先 `docker compose down -v` 清掉索引卷再 `up`。

## 让 HalGuard「随 AI 一起运行」

HalGuard 是常驻代理，最可靠的做法是与本地 LLM 一起启动。任选其一：

- **Linux/macOS**：用 `systemd` / `launchd` / `supervisor` 同时拉起 `ollama serve`（或你的推理服务）与 `halguard serve`。
- **Windows**：把两个程序放进同一个开机/启动脚本，或用 `nssm` 注册为服务。
- **Docker**：见上方「一键部署（Docker Compose）」；默认编排里也带了可选的 Ollama 服务（`--profile ollama`）。

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
| `claim_extraction` | `heuristic` | 断言抽取方式：`heuristic`（规则）/ `llm`（用后端模型抽取，更准但更慢）|
| `append_warning` | `true` | flag 时是否在回复末尾追加警告 |
| `websearch_enabled` | `false` | **联网搜索开关**：开启后每次提问先搜索网页，并强制模型仅依据搜索结果作答（环境变量 `HALGUARD_WEBSEARCH=1`；请求体传 `"websearch": true/false` 可按次覆盖）|
| `websearch_max_results` | `5` | 每次搜索注入的网页结果数（`HALGUARD_WEBSEARCH_MAX_RESULTS`）|
| `websearch_timeout` | `10.0` | 单次搜索超时秒数（`HALGUARD_WEBSEARCH_TIMEOUT`）|
| `websearch_min_max_tokens` | `4096` | 联网模式 `max_tokens` 下限：思考模型推理会耗尽小预算导致**空正文**，服务端会自动抬高预算，并在"正文空但有推理内容"时**双倍预算重试一次** |

### 联网搜索模式（强制接地）

打开开关后，代理的流程变为：**搜索 → 注入 → 生成 → 验证**——

1. 以用户问题为查询词执行网络搜索（搜索源依次回退：**360 搜索 → Bing → DuckDuckGo**，均为无需 API Key 的 HTML 解析）；
2. 搜索结果作为[网络搜索结果]注入 system 消息，并强制指令"仅依据搜索结果作答、标注来源编号、搜不到就明说，不要编造"；
3. 回答中的断言会与搜索结果做语义支持度校验，低支持度断言照常被标记/告警；
4. 内置**相关性门控**：与查询完全无关的搜索结果（Bing 对程序化请求的反爬污染特征是 HTTP 200 但结果无关）会被自动剔除，防止错误上下文误导模型。

搜索失败时不会沉默失败：会向模型注入"搜索失败"的诚实指令（要求声明未经联网核实、不得编造易变事实）。

```bash
# 请求级开关示例（OpenAI 兼容接口，额外字段 websearch）
curl http://localhost:8849/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"今天有什么科技新闻？"}],"websearch":true}'
```

内置网页客户端 `chat.html` 顶栏已有「🌐 联网搜索」开关，回答下方会显示搜索来源列表。

> **思考模型提示**：qwen3 等思考模型的推理会消耗 token 预算，联网模式上下文更长。
> HalGuard 服务端已内置兜底：联网模式自动把 `max_tokens` 抬到
> `websearch_min_max_tokens`（默认 4096），若仍出现"正文空、只有推理内容"会**自动双倍预算重试一次**——
> 因此即使客户端传了很小的 `max_tokens`，联网模式下模型也能正常回答。

## 局限与说明

- 验证依赖「检索到的上下文」作为事实基准。**知识库越准、越全，抗幻觉效果越好**。
- TF-IDF 嵌入是关键词级，语义泛化弱；生产环境务必安装 `sentence-transformers` 获得句级语义嵌入。
- 本工具是**降低**而非**消除**幻觉：它把「无依据的编造」暴露在风险评分中，并尽量约束模型「有依据才说」。

## License

MIT

# HalGuard — 本地 LLM/SLM 抗幻觉中间件镜像
# 构建包含核心依赖 + 高质量语义嵌入（sentence-transformers + faiss）
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装包（利用 Docker 层缓存：代码不变就不重装依赖）
# 使用清华 PyPI 镜像加速国内构建（海外网络构建可换回官方源）
COPY pyproject.toml README.md ./
COPY halguard ./halguard
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple ".[quality]"

# 知识库默认目录（运行时可用卷覆盖）
COPY knowledge ./knowledge
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 运行时配置（均可被 HALGUARD_* 环境变量覆盖）
# - 嵌入模型不打入镜像：挂载宿主机 ./halguard_models 到 /models，避免国内拉取 HuggingFace 失败
# - 后端 LLM 默认指向宿主机 Ollama（host.docker.internal）
ENV HALGUARD_KB_PATH=/app/knowledge \
    HALGUARD_INDEX_PATH=/data/index \
    HALGUARD_ST_MODEL=/models/paraphrase-multilingual-MiniLM-L12-v2 \
    HALGUARD_BACKEND_BASE_URL=http://host.docker.internal:11434/v1 \
    HALGUARD_BACKEND_MODEL=llama3.1:8b \
    HALGUARD_LISTEN_HOST=0.0.0.0 \
    HALGUARD_LISTEN_PORT=8849

# /data 持久化检索索引；/models 挂载本地嵌入模型
VOLUME ["/data", "/models"]
EXPOSE 8849

ENTRYPOINT ["/entrypoint.sh"]

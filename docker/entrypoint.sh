#!/bin/sh
# HalGuard 容器入口：首次启动自动摄入知识库，然后启动代理服务
set -e

INDEX_DIR="${HALGUARD_INDEX_PATH:-/data/index}"
KB_DIR="${HALGUARD_KB_PATH:-/app/knowledge}"

# 索引不存在（首次启动或索引被清空）时自动构建
if [ ! -f "${INDEX_DIR}/store.pkl" ]; then
    echo "[entrypoint] 未检测到检索索引，正在摄入知识库 ${KB_DIR} ..."
    halguard ingest "${KB_DIR}"
fi

exec halguard serve

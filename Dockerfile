# ScholarLens · 学术文献 Agentic RAG 系统 — 容器镜像
FROM python:3.11-slim

WORKDIR /app

# 系统依赖(构建原生扩展用)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖(单独一层,利用缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码
COPY . .

# Chainlit 前端端口
EXPOSE 8000

# 通过 --env-file 注入 OPENAI_API_KEY / OPENAI_BASE_URL,绝不硬编码
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

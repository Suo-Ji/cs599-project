<div align="center">

# 📚 ScholarLens · 学术文献 Agentic RAG 系统

**基于 LangGraph 的自我修正式学术问答系统 —— 混合检索 · 交叉编码器重排 · 零幻觉生成**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-green.svg)](https://github.com/langchain-ai/langgraph)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-V4--Flash-purple.svg)](https://siliconflow.cn)
[![Tests](https://img.shields.io/badge/tests-54%20passed-brightgreen.svg)](#测试)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

</div>

---

## 项目简介

> 让大模型只基于真实 arXiv 论文回答学术问题 —— 自动检索、重写查询、交叉编码器精排、生成并自我核验,拒绝幻觉,逐句可溯源。

传统 RAG 直接把向量检索结果丢给 LLM,在**密集科学文献**(复杂术语、数学公式、RMSE/PICP 等评估指标)上常出现召回不准、答案编造、无法溯源三大痛点。本项目用 **Agentic 工作流**解决这些问题:

- 🔍 **混合检索**:稠密向量(语义)+ BM25(精确术语/数值)双路召回,经 **RRF 融合**取长补短
- 🎯 **交叉编码器重排**:`bge-reranker-large` 精排,只把最相关的 Top-5 喂给 LLM
- 🧠 **自我修正循环**:检索不够 → 自动重写查询(最多 3 轮);答案不实 → 触发幻觉检查并重新生成(最多 2 轮)
- 🛡️ **零幻觉**:每句回答都来自检索文档,忠实度量化打分,违规则回炉
- 📎 **引用溯源**:每个结论标注来源文件 + 页码
- 💬 **Chainlit 聊天界面**:流式回答、可折叠推理过程、中文友好

### 方向

> **方向一:Agentic AI 原生开发**

本项目从零设计为 Agent 架构 —— 用 LangGraph 把"检索-评估-重写-生成-核验"编排成**带状态、带循环、带条件分支**的状态图,而非简单的线性 RAG 管道。Agent 能自主决定何时重查、何时重写、答案是否可信,具备真正的自主推理与自我修正能力。

---

## ✨ 核心特性

| 能力 | 实现 |
|---|---|
| 🧩 语义分块 | 按章节/逻辑段落切分(500-800 token),**公式与表格永不拆分** |
| 🔀 混合检索 | Dense(余弦) + Sparse(BM25),`RRF(k=60)` 融合 |
| 🎖️ 重排精排 | `BAAI/bge-reranker-large` 交叉编码器,Top20 → Top5 |
| 🔁 自我修正 | 查询重写循环 + 幻觉核验循环,双重防死循环 |
| 📊 自动评估 | LLM-as-judge 打分:忠实度 / 上下文召回 / 答案相关性 / 精确率 |
| 🗂️ 真实语料 | 内置 7 篇真实 arXiv PDF(跨 cs.LG/RO/stat.ML/CV/physics 五领域) |
| 🌐 arXiv 抓取 | 标准库实现,断点续抓,自动补全元数据 |

---

## 🛠️ 技术栈

| 类别 | 技术 |
|---|---|
| **AI IDE** | Trae CN |
| **LLM** | DeepSeek API(经硅基流动 SiliconFlow 接入,`deepseek-ai/DeepSeek-V4-Flash`) |
| **Agent 框架** | LangGraph + LangChain(状态图 · 条件边 · 自我修正) |
| **向量库** | ChromaDB |
| **嵌入/重排** | sentence-transformers(`bge-large-zh-v1.5` · `bge-reranker-large`) |
| **稀疏检索** | rank_bm25(Okapi BM25) |
| **数据模式** | Pydantic v2 |
| **PDF 解析** | pypdf |
| **前端** | Chainlit(流式聊天 · 推理步骤) |
| **容器** | Docker(见下方 `Dockerfile`) |
| **评估** | LLM-as-judge + 确定性启发式指标 |
| **测试** | pytest(54 项) |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  前端  Chainlit 聊天界面 (app.py)                            │
│  流式问答 · 可折叠推理步骤 · 参考来源引用                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  Layer 3  Agentic 控制流 (src/agent/)  LangGraph 状态图     │
│  Query_Analyzer → Retrieve → Document_Grader                │
│        ↕ 重写循环(上限3)        ↓                          │
│  Generate_Answer → Hallucination_Checker → END              │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 2  混合检索与重排 (src/retrieval/)                   │
│  Dense(余弦) + Sparse(BM25) → RRF融合(k=60)              │
│  → 交叉编码器重排(bge-reranker-large) → 最终 Top5         │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 1  文档处理 (src/ingestion/)                         │
│  PDF解析(保留结构) → 语义分块(500-800 token,10%重叠)    │
│  → 元数据注入 → 向量索引(Chroma + bge-large-zh)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 目录结构

```
RAG/
├── app.py                       # Chainlit 聊天前端(流式问答 + 推理步骤)
├── main.py                      # 环境自检入口
├── config/config.yaml           # 全部可调参数集中于此
├── data/
│   ├── pdfs/                    # 7 篇真实 arXiv PDF 语料
│   └── arxiv_testset.json       # arXiv 抓取元数据清单
├── src/
│   ├── common/                  # 配置加载、日志、Pydantic 数据模式
│   ├── ingestion/               # Layer 1: 解析 → 语义分块 → 向量索引
│   │   ├── parser.py            #   PDF 解析(章节/公式/表格结构)
│   │   ├── splitter.py          #   语义分块(原子保护公式表格)
│   │   ├── embedder.py          #   bge 嵌入
│   │   └── vectorstore.py       #   Chroma 向量库
│   ├── retrieval/               # Layer 2: 混合检索与重排
│   │   ├── bm25_retriever.py    #   BM25 稀疏检索
│   │   ├── rrf.py               #   倒数排序融合(纯函数)
│   │   ├── reranker.py          #   交叉编码器重排
│   │   └── hybrid.py            #   混合引擎集成
│   ├── agent/                   # Layer 3: LangGraph 自我修正控制流
│   │   ├── graph.py             #   状态图(6 节点 + 条件边 + 重试上限)
│   │   ├── llm_client.py        #   LLM 网关(重试/超时/离线降级)
│   │   └── prompts.py           #   节点系统提示词(确定性二元判断)
│   └── evaluation/              # Layer 4: 评估
│       ├── judge.py             #   LLM-as-judge
│       └── metrics.py           #   确定性启发式评分
├── tests/                       # 54 项单元测试 + 评估管道
└── scripts/                     # arXiv 抓取 / 演示脚本
```

---

## 🚀 环境搭建

### 前置要求

- **Python ≥ 3.10**
- **DeepSeek / 硅基流动 API Key**(在线生成必需;无 key 也可离线体验全部控制流)

### 1. 依赖安装

```bash
git clone <your-repo-url>
cd RAG

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### 2. 环境变量配置(⚠️ 不硬编码 API Key)

在项目根目录创建 `.env` 文件(已被 `.gitignore` 忽略,key 永不进版本库):

```bash
# .env
OPENAI_API_KEY=sk-你的硅基流动key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
```

> 为什么是 `OPENAI_*`?系统用 OpenAI 兼容接口接入,**硅基流动 / DeepSeek / Ollama / vLLM** 任一兼容服务都只需改这两个变量 + `config.yaml` 的 `model` 名,**零代码改动**。代码中绝不硬编码 key —— 全部走环境变量。

模型名在 `config/config.yaml` 配置:
```yaml
agent:
  llm:
    model: "deepseek-ai/DeepSeek-V4-Flash"   # 当前默认
```

### 3. 启动步骤

**方式一:本地直接运行**

```bash
# 验证在线连接(应显示 status: ONLINE)
python scripts/check_llm_online.py

# 启动 Chainlit 聊天前端
chainlit run app.py
# 浏览器自动打开 http://localhost:8000
```

**方式二:Docker 运行**

```bash
docker build -t scholarlens .
docker run -p 8000:8000 --env-file .env scholarlens
```

### 4. 验证安装

```bash
python -m pytest tests/unit/ -q        # 54 项测试应全部通过
python tests/evaluation/run_eval_real.py  # 在 7 篇真实论文上评估
```

---

## 📖 使用方法

启动 Chainlit 后,在聊天框直接提问(中英文均可,回答默认中文):

```
🧑 "层次化优势加权方法是怎么用于在线强化学习微调的?"
🤖 直接回答 → 拓展(动机/机制/实验数据)→ 局限性 + 参考来源引用

🧑 "Which paper solves posterior score estimation for linear inverse problems?"
🤖 (自动定位到 2606.17048 并生成中文回答)
```

### 其他入口

| 命令 | 作用 |
|---|---|
| `python scripts/demo_agent.py` | 终端看 Agent 决策追踪 |
| `python scripts/demo_retrieval.py` | 混合检索演示 |
| `python scripts/validate_ingestion.py` | 注入流程元数据分布 |
| `python scripts/fetch_arxiv_testset.py` | 从 arXiv API 抓取更多论文 |
| `python tests/evaluation/run_eval_real.py` | 真实文献评估 |

---

## 📊 实验结果

在 7 篇真实 arXiv 论文(跨 5 领域)上的评估:

| 指标 | 得分 | 说明 |
|---|---|---|
| **检索命中准确率** | **10/10** | 10 个查询的 Top-1 召回全部来自正确论文 |
| 忠实度(Faithfulness) | 1.000 | 零幻觉,答案严格基于上下文 |
| 答案相关性 | 1.000 | 准确解决用户问题 |
| 上下文召回 | 0.886 | 检索到回答所需的绝大部分信息 |
| 上下文精确率 | 1.000 | 检索结果高度相关 |

> 跨 cs.LG / cs.RO / stat.ML / cs.CV / physics 五领域,BM25+RRF 仍能 10/10 命中正确论文。

---

## 🧪 测试

```bash
python -m pytest tests/unit/ -v
```

**54 项单元测试**覆盖:RRF 融合数学、语义分块(公式/表格原子保护)、混合检索集成、LangGraph 条件边与重试上限、容错降级、评估指标与判定回退。

---

## 🔬 核心设计亮点

1. 原子块保护(公式/表格永不拆分)

解析器将文本分类为**原子块**(equation/table)与段落。分块器保证公式(`$$...$$`、`\begin{equation}`)与表格完整保留在单个 chunk 内,重叠(overlap)绝不复制原子块。这是处理数学密集文献的关键。

2. 双重防死循环的自我修正

Agent 含两个独立重试边界:① Document_Grader 判定 irrelevant → Query_Rewriter 重写(上限 3 轮),达上限强制生成;② Hallucination_Checker 判定不实 → 重新生成(上限 2 轮),达上限回退重新检索。所有 LLM/检索调用包裹在 try-except 中,**token 超时永不崩溃执行图**。

3. 确定性二元判断

Document_Grader 与 Hallucination_Checker 输出 `score + decision`,score 低于阈值时**强制翻转** decision,确保条件边基于稳定的机器可读信号分支,而非 LLM 主观措辞。

4. 离线/在线零代码切换


无 API key 时,系统进入确定性离线模式(完整演示控制流);填入 key 自动切到真实 LLM。同一套脚本,一个开关。

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码
COPY . .

# Chainlit 前端端口
EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📌 项目状态

- [x] **Proposal** —— 需求分析与架构设计
- [x] **MVP** —— 4 层架构 + Chainlit 前端 + 真实文献验证
- [x] LangGraph 自我修正控制流(查询重写 + 幻觉核验)
- [x] 混合检索 + RRF + 交叉编码器重排
- [x] arXiv API 真实语料抓取与评估(检索 10/10 命中)
- [x] Chainlit 聊天界面(流式 + 中文回答 + 引用溯源)
- [x] Docker 容器化
- [ ] **Final** —— 接入真实 bge 嵌入向量库端到端基准 · 多轮对话上下文

---

## 📝 License

MIT License — 仅供学习与研究使用。

---

<div align="center">

**如果这个项目对你有帮助,欢迎 ⭐ Star**

Built with ❤️ using LangGraph · DeepSeek · Chainlit

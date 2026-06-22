# Architecture · 详细架构说明

> ScholarLens · 学术文献 Agentic RAG 系统
> 本文是面向开发者的**深度架构参考**,覆盖分层设计、LangGraph 状态机、组件契约、数据流、配置模型与设计取舍。
> 配合 `项目报告.md`(答辩视角)与 `README.md`(使用视角)阅读。

---

## 目录

1. [架构总览](#1-架构总览)
2. [分层职责与依赖拓扑](#2-分层职责与依赖拓扑)
3. [Layer 1 · 文档处理管道](#3-layer-1--文档处理管道)
4. [Layer 2 · 混合检索与重排引擎](#4-layer-2--混合检索与重排引擎)
5. [Layer 3 · LangGraph Agentic 控制流](#5-layer-3--langgraph-agentic-控制流)
6. [Layer 4 · 评估管道](#6-layer-4--评估管道)
7. [数据契约(Pydantic Schema)](#7-数据契约pydantic-schema)
8. [配置模型](#8-配置模型)
9. [全链路数据流](#9-全链路数据流)
10. [关键设计决策与取舍](#10-关键设计决策与取舍)
11. [目录结构](#11-目录结构)
12. [扩展指南](#12-扩展指南)

---

## 1. 架构总览

系统采用 **4 层分层架构 + 顶部 Chainlit 前端**,严格单向依赖。每一层是一个可独立测试、可替换的模块。

```
┌─────────────────────────────────────────────────────────────┐
│  前端  Chainlit (app.py) — 流式问答 · 推理步骤 · 引用溯源    │
└──────────────────────────────────┬──────────────────────────┘
                                   │ user message / streamed answer
┌─────────────────────────────────────────────────────────────┐
│  Layer 3   Agentic 控制流  (src/agent/)    LangGraph 状态图  │
│  6 节点 + 2 条件边 + 双重防死循环(重写≤3 / 生成≤2)        │
└──────────────────────────────────┬──────────────────────────┘
                                   │ processed_query → Top5 chunks
┌─────────────────────────────────────────────────────────────┐
│  Layer 2   混合检索与重排  (src/retrieval/)                 │
│  Dense(余弦) + Sparse(BM25) ──RRF(k=60)──▶ Cross-Encoder▶Top5│
└──────────────────────────────────┬──────────────────────────┘
                                   │ candidate chunks
┌─────────────────────────────────────────────────────────────┐
│  Layer 1   文档处理  (src/ingestion/)                       │
│  PDF 解析(保留结构) → 语义分块(500-800 tok, 10%重叠)    │
│  → 元数据注入 → 向量索引(Chroma + bge-large-zh)          │
└──────────────────────────────────┬──────────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │  common/  配置 · 日志 · Pydantic 数据契约    │  ← 全局共享
            └─────────────────────────────────────────────┘
```

**设计原则**
- **单向依赖**:`common ← ingestion ← retrieval ← agent ← evaluation`,绝不反向引用。
- **契约驱动**:层间通过 Pydantic 模型传递数据,接口强类型。
- **注入式组件**:检索器、LLM、重排器均可注入(`__init__` 参数),便于测试与替换。
- **配置即代码**:`config/config.yaml` 是唯一参数源,代码零硬编码。

---

## 2. 分层职责与依赖拓扑

```
common   (config.py / logging_setup.py / schemas.py)
   ▲
ingestion (parser · splitter · keyword_extractor · tokenizer_utils
           embedder · vectorstore · pipeline)
   ▲
retrieval (bm25_retriever · rrf · reranker · hybrid)
   ▲
agent     (llm_client · prompts · graph)
   ▲
evaluation(dataset · metrics · judge)
```

| 层 | 输入 | 输出 | 主要组件 |
|---|---|---|---|
| **common** | — | `AppConfig` / 日志 / Schema | 配置单例、rich 日志、Pydantic 模型 |
| **ingestion** | PDF/文本路径 | `list[DocumentChunk]` | 解析器、语义分块器、嵌入器、向量库 |
| **retrieval** | 查询字符串 | `list[(DocumentChunk, score)]` | BM25、RRF、Cross-Encoder、Hybrid |
| **agent** | 用户问题 | `AgentState`(含生成答案) | LLM 网关、提示词、LangGraph |
| **evaluation** | 查询+答案+上下文 | `EvaluationReport` | 数据集、评分器、LLM-as-judge |

---

## 3. Layer 1 · 文档处理管道

> 模块:`src/ingestion/` · 职责:把原始 PDF 转成结构化、可检索的语义块。

### 3.1 解析器 `parser.py`

```
parse_document(path: str | Path) -> ParsedDocument
```

**结构化解析**(保留文档结构):
- 支持 `.pdf`(pypdf 逐页提取)、`.txt`/`.md`;其他扩展名抛 `ValueError`。
- **标题检测**:正则识别编号标题(`1 Introduction` / `2.1 Loss`,层级由点数决定)、关键词标题(Abstract/Methodology/...)、全大写短行。
- **原子块分类**:文本被归类为 `TextBlock`,其中:
  - **equation**:`$$...$$`、`\[...\]`、`\begin{equation|align|gather|eqnarray|multline}`
  - **table**:`\begin{table|tabular|longtable}`、连续 Markdown `|...|` 行
  - **paragraph**:其余(行内 `$...$` 保留在段落内)
- **代理字符清理**:`sanitize_text()` 在提取点统一剔除 pypdf 产出的孤立代理单元(数学字母 `\ud835` 等),保证后续 `md5`/编码安全。
- **容错**:单页/单文件提取失败不中断整批。

```
ParsedDocument { source_file, page_count, metadata, sections[] }
  └─ ParsedSection { title, level, page_start, page_end, blocks[] }
       └─ TextBlock { kind: paragraph|equation|table|heading, text, page_number }
```

### 3.2 语义分块器 `splitter.py`

```
SemanticSplitter(config, token_counter=None).split_document(doc) -> list[DocumentChunk]
```

**核心契约**(目标 token 来自 config):
- 块目标 500-800 token、相邻块 10% 重叠。
- **公式与表格永不拆分**:原子块作为一个整体单元,即使超过窗口也保留完整(仅记 warning)。
- **超大段落按句子切分**(正则 `(?<=[.!?])\s+(?=[A-Z0-9])`),**绝不字符级截断**。
- **重叠逻辑**:`_compute_overlap` 反向遍历尾部非原子单元直到累计达 `_overlap_target`;**原子块永不作为 overlap 复制**(避免公式/表格在两个块重复)。
- **chunk_id**:`<source_stem>::<index:04d>::<md5前10位>`(确定性、内容寻址)。

### 3.3 关键词注入 `keyword_extractor.py`

```
extract_keywords(text, config) -> list[str]
```
- 词典匹配:`config.ingestion.keyword_lexicon` 的 `metrics`(RMSE/NLL/PICP/MPIW...)+ `models`(BNN/GNN/MLP/EDL...)。
- 通用缩写兜底:`[A-Z][A-Z0-9]{1,6}`,负向先行断言避免匹配连字符复合词前缀(如 "MC-Dropout" 不重复抽 "MC")。

### 3.4 嵌入与向量库

- **`embedder.py`**:`Embedder(model_name, device=None, normalize=True)`,`encode(texts)->np.ndarray (N,dim)`、`encode_one(text)->1-D`。sentence-transformers 懒加载。
- **`vectorstore.py`**:`VectorIndex(persist_dir, collection_name, embedder, distance_metric="cosine")`。
  - `add_chunks(chunks, batch_size=64) -> int`(批量嵌入 + upsert,按 `chunk.id` 幂等)
  - `query(query_text, top_k=20, where=None) -> list[(chunk, score)]`,`where` 透传 **Chroma 元数据标量过滤**(`source_file`/`section_title`/`page_number`)
  - `score = max(0, 1 - distance)`(cosine 距离 [0,2] → 相似度 [0,1])
  - `count()` / `reset()`

---

## 4. Layer 2 · 混合检索与重排引擎

> 模块:`src/retrieval/` · 职责:融合双路召回,精排出 Top-5。

### 4.1 编排流程(`hybrid.py`)

```
query
 ├─ _dense_search  → VectorIndex.query(top_k=dense_top_k, where=where)   # 稠密
 ├─ _sparse_search → BM25Retriever.query(top_k=sparse_top_k)             # 稀疏
 │
 └─ _fuse:  fuse_scores([dense_ids, sparse_ids], k=rrf_k, top_n=rerank_candidates)
            ↓
          融合候选 Top20
            ↓
          (可选) CrossEncoderReranker.rerank(query, candidates, top_n=final_top_n)
            ↓
          Top5 → 返回
```

**`HybridRetriever` 接口**
```
HybridRetriever(config, vector_index=None, bm25=BM25Retriever(), reranker=None)
.retrieve(query, where=None, final_top_n=None) -> list[(DocumentChunk, float)]
.for_corpus(chunks, ...) -> HybridRetriever   # 类方法,快速构建
.index_chunks(chunks) -> None
```

> **关键细节**
> - `reranker=None` 是**合法配置**(禁用重排,直接返回 RRF 序),非异常。
> - `where` **仅作用于 dense 检索**;BM25 不支持元数据过滤。
> - 实际调用的是 `fuse_scores`(带 `top_n` 截断的 `reciprocal_rank_fusion` 包装)。

### 4.2 稀疏检索 BM25(`bm25_retriever.py`)

```
tokenize(text) -> list[str]   # 正则 [A-Za-z0-9]+(?:\.[0-9]+)?,小写化,滤停用词
BM25Retriever.add_chunks(chunks) -> int        # 替换式重建
BM25Retriever.query(query, top_k=20) -> list[(chunk, score)]
BM25Retriever.ranked_ids(query, top_k=20) -> list[str]   # 供 RRF 消费
```
- 基于 `rank_bm25.BM25Okapi`;分数为未归一化正浮点,**仅用于排序**(RRF 消费的是 rank,不是分数)。

### 4.3 RRF 融合(`rrf.py`)— 纯函数

```
DEFAULT_RRF_K = 60
reciprocal_rank_fusion(ranked_lists, k=60) -> list[(id, score)]
    RRF_Score(d) = Σ_m  1 / (k + r_m(d))
fuse_scores(ranked_lists, k=60, top_n=None) -> list[(id, score)]
```
- 文档在某方法缺席 → 该项贡献 0。
- **确定性**:平局按首次出现顺序(first_seen)打破。
- 无 I/O、无状态,可独立单测(TDD 优先)。

### 4.4 交叉编码器重排(`reranker.py`)

```
CrossEncoderReranker(model_name, device=None)
.rerank(query, candidates, top_n=5) -> list[(chunk, score)]
.score_pair(query, passage) -> float   # 单对评分,不兜底(会真实加载模型)
```
- `rerank` 跳过空候选;模型缺失/打分异常 → **优雅降级**,按原序截断返回。
- 与 `score_pair` 行为不同:`score_pair` 不捕获异常(debug/eval 用)。

---

## 5. Layer 3 · LangGraph Agentic 控制流

> 模块:`src/agent/` · 职责:状态驱动的自我修正工作流。

### 5.1 LLM 网关(`llm_client.py`)

所有外部 LLM 调用**唯一入口**,把 provider 屏蔽在 Agent 之外:
```
LLMClient(config, api_key=None, base_url=None, client=None)
  ├─ offline 属性: client is None and not api_key
  ├─ invoke(system, user) -> str          # 重试 + 指数退避,终态抛 LLMError
  ├─ invoke_json(system, user) -> dict    # 容忍 markdown fence 的 JSON 解析
  └─ 模块级 load_dotenv(): 导入时加载 .env
```
- **离线模式**(无 `OPENAI_API_KEY`):返回确定性 canned JSON,匹配各节点 schema,使图无网络也能测试。
- **重试**:`range(1, max_retries+2)`,退避 `min(2**(attempt-1), 8)` 秒。
- **provider 无关**:用 OpenAI 兼容接口,`.env` 的 `OPENAI_BASE_URL` 切换 SiliconFlow/OpenAI/Ollama/vLLM。

### 5.2 提示词(`prompts.py`)

5 个 system prompt 常量,均为客观、确定性语气(无比喻/拟人):
`QUERY_ANALYZER_SYSTEM` · `QUERY_REWRITER_SYSTEM` · `DOCUMENT_GRADER_SYSTEM` · `GENERATE_ANSWER_SYSTEM` · `HALLUCINATION_CHECKER_SYSTEM`

- **确定性二元判断**:Grader/Checker 输出 `score + decision`,score 低于阈值时节点**强制翻转** decision → 条件边基于稳定机器可读信号分支。
- **中文回答**:`GENERATE_ANSWER_SYSTEM` 强制简体中文,术语/数值保留原文。
- **中→英检索**:Analyzer/Rewriter 强制检索词写英文(匹配英文语料)。

### 5.3 状态图(`graph.py`)

#### 节点(6 个,均在 `AgentNodes` 类中)

| 节点 | 职责 | 返回状态增量 |
|---|---|---|
| `query_analyzer` | 意图分类 + 优化查询(中→英) | `processed_query`, `query_route` |
| `retrieve` | 调用 HybridRetriever | `retrieved_documents`, `retrieval_scores` |
| `document_grader` | 二元相关性评估 | `grade_decision` |
| `query_rewriter` | 重写查询(词汇失配修复) | `processed_query`, `retry_count+1` |
| `generate_answer` | 生成中文拓展答案 | `generation`, `generation_attempts+1` |
| `hallucination_checker` | 二元忠实度核验 | `hallucination_decision` |

#### 边与条件路由

```
START → query_analyzer → retrieve → document_grader
                                     │
                ┌────────────────────┼─ route_after_grading ─────────────┐
                │                     │                                    │
        relevant                   irrelevant                          irrelevant
                │                     │ (< max_rewrites)                (≥ max_rewrites)
                │                     ▼                                    │
                │            query_rewriter → retrieve                    │
                │                     (重写循环,上限 3)                  │
                ▼                                                          │
            generate ◀────────────────────────────────────────────────────┘
                │
                ▼
       hallucination_checker
                │
        ┌───────┴────────── route_after_hallucination ──────────┐
        │                  │                                      │
     grounded         not_grounded                            not_grounded
        │                  │ (< gen_limit)                     (≥ gen_limit)
        ▼                  ▼                                      ▼
       END            generate(重生,上限 2)                     retrieve(回退)
```

**路由函数返回值 → 节点映射**
```
route_after_grading:
    relevant                                    → "generate"
    irrelevant & retry_count < max_rewrites     → "rewrite"  (→ query_rewriter)
    irrelevant & 达上限                          → "generate" (强制生成)

route_after_hallucination:
    grounded                                    → "end"   (→ END)
    not_grounded & attempts < gen_limit         → "generate"
    not_grounded & 达上限                        → "retrieve"  (回退重检)
```

#### 双重防死循环
- **重写循环**:`config.agent.max_rewrite_retries = 3`(查询重写上限)
- **生成循环**:`generation_attempt_limit = 2`(幻觉核验重生上限)—— 两个**独立**上限。
- 任一达上限 → 输出尽力答案并终止,**绝不无限循环**。

#### 状态 reducer(`GraphState` TypedDict)

| 字段 | reducer | 语义 |
|---|---|---|
| `question`/`processed_query`/`query_route`/`generation`/`retry_count`/`generation_attempts`/`grade_decision`/`hallucination_decision`/`retrieved_documents`/`retrieval_scores` | `_overwrite` | 取最新值 |
| `evaluation_logs` | `_append` | 追加(完整决策追踪) |

#### 容错降级

所有节点 LLM/检索调用经 `_safe_json` / `_safe_text` 包裹:
```
_safe_json(system, user, default): 任意异常 → 返回 default(安全默认值)
_safe_text(system, user):          任意异常 → 返回 "Answer unavailable..."
```
**token 超时永不崩溃执行图**——节点返回降级状态,图继续运行。

#### 公开 API
```
build_graph(config, retriever, llm, generation_attempt_limit=2) -> CompiledGraph
run_agent(question, config, retriever, llm, generation_attempt_limit=2) -> AgentState
```

---

## 6. Layer 4 · 评估管道

> 模块:`src/evaluation/` · 职责:LLM-as-judge 量化评估。

### 6.1 评分器(`metrics.py`)— 确定性启发式

```
score_faithfulness(answer, contexts) -> float     # 被支撑句子占比(≥半数 key term 命中)
score_context_recall(ground_truth, contexts) -> float   # GT key term 在上下文命中比
score_answer_relevance(question, answer) -> float       # |Q∩A| / |Q|
score_context_precision(retrieved, ground_truth) -> float # 与 GT 共享 key term 的块占比
extract_key_terms(text) -> frozenset[str]  # @lru_cache;数字+缩写+长词
```
全部返回 [0,1] 纯函数,无 I/O,可独立单测。

### 6.2 LLM-as-judge(`judge.py`)

```
LLMJudge(config, llm=None)
  ├─ faithfulness(answer, contexts) -> float
  ├─ context_recall(ground_truth, contexts) -> float
  ├─ answer_relevance(question, answer) -> float
  └─ context_precision(retrieved, ground_truth) -> float   # 始终走启发式,不经 LLM
```
- 在线 → LLM JSON 评分(键 `faithfulness`/`context_recall`/`answer_relevancy`);离线/失败 → 回落 `metrics.*` 启发式。
- `_clip` 限制到 [0,1]。

### 6.3 数据集(`dataset.py`)

```
load_dataset(path=None, config=None) -> EvalDataset
EvalQuery { id, category, query, ground_truth, key_terms, relevant_section, target_arxiv_id }
EvalDataset { description, source_paper, queries[] }
```

---

## 7. 数据契约(Pydantic Schema)

> `src/common/schemas.py` — 全系统强类型契约。

| 模型 | 用途 | 关键字段 |
|---|---|---|
| `DocumentChunk` | 检索/存储单元 | `id, content, source_file, section_title, page_number, page_range, keywords, chunk_index, token_count, metadata` + `to_filterable_metadata()` |
| `AgentState` | LangGraph 对外状态 | `question, processed_query, query_route, retrieved_documents, retrieval_scores, generation, retry_count, generation_attempts, grade_decision, hallucination_decision, evaluation_logs` |
| `QueryAnalysis` | Router 输出 | `route: default\|methodology\|experimental\|definition, processed_query, needs_definition` |
| `DocumentGrade` | Grader 输出 | `score[0,1], decision: relevant\|irrelevant, rationale` |
| `HallucinationGrade` | Checker 输出 | `grounded_fraction[0,1], decision: grounded\|not_grounded, unsupported_claims` |
| `EvaluationResult` | 单查询评估 | `query, answer, ground_truth, contexts, faithfulness, answer_relevancy, context_recall, context_precision, passed, notes` |
| `EvaluationReport` | 聚合报告 | `results[], mean_*, total_queries` |

> ⚠️ **命名易错点**:Schema 字段用 `answer_relevancy`(末尾 y),而 `metrics` 函数名是 `score_answer_relevance`(末尾 ce)。两者指同一指标。

---

## 8. 配置模型

> `src/common/config.py` + `config/config.yaml`

```
AppConfig
 ├─ paths        { pdf_dir, vectorstore_dir, collection_name, eval_dataset, eval_results }
 ├─ ingestion    { embedding_model, embedding_dim, chunk_min/max_tokens, overlap_ratio,
 │                 min_chunk_tokens, keyword_lexicon{metrics[], models[]} }
 ├─ vectorstore  { distance_metric, allow_reset, hnsw{...} }
 ├─ retrieval    { dense_top_k, sparse_top_k, rrf_k, rerank_model,
 │                 rerank_candidates, final_top_n, grade_score_threshold }
 ├─ agent        { llm{model, temperature, request_timeout, max_retries},
 │                 max_rewrite_retries, grading_binary, hallucination_binary,
 │                 grounded_claim_threshold }
 ├─ evaluation   { metrics[], llm_as_judge_model, fail_on{...} }
 └─ logging      { level, rich_console }
```

- **类型化 + 防拼写错误**:所有 section `extra="forbid"`,配置拼错即报错。
- **单例**:`get_config()` 经 `@lru_cache` 缓存,全系统共享。
- **路径属性**:`pdf_dir` / `vectorstore_dir` / `eval_dataset_path` / `eval_results_path` 基于 `PROJECT_ROOT` 解析。
- **密钥隔离**:API key 走 `.env` 环境变量(`load_dotenv`),**绝不进 config.yaml / 版本库**。

---

## 9. 全链路数据流

```
[真实 arXiv PDF]
   │ pypdf 提取(章节/页码) + sanitize_text(剔除代理字符)
   ▼
[TextBlock: paragraph | equation | table]   ← 原子块,公式/表格不拆
   │ SemanticSplitter(500-800 tok, 10% 重叠, 原子保护)
   ▼
[DocumentChunk + 元数据(keywords/section/page/chunk_index)]
   │ to_filterable_metadata()
   ├──▶ BM25Retriever.add_chunks()      # 内存索引
   └──▶ VectorIndex.add_chunks()        # (可选)Chroma 向量
   ▼
[用户问题] (中文)
   │ query_analyzer: 中→英检索词 + 意图分类
   ▼
HybridRetriever.retrieve(processed_query)
   ├─ dense: VectorIndex.query(top_k=20)
   ├─ sparse: BM25Retriever.query(top_k=20)
   └─ fuse_scores([ids], k=60) → Top20
        │ CrossEncoderReranker.rerank → Top5
        ▼
   [5 个候选块]
   │ document_grader: score + decision(阈值翻转)
   ├─ irrelevant & <3 → query_rewriter → 重新 retrieve
   ▼ relevant / 达上限
   generate_answer: 简体中文三段式(直接回答/拓展/局限性)+ [来源]
   │ hallucination_checker: grounded_fraction + decision
   ├─ not_grounded & <2 → 重新 generate
   ├─ not_grounded & 达上限 → 重新 retrieve
   ▼ grounded / 达上限
   [可信答案 + 参考来源引用]
   │ _format_final_sources_from_state: 提取 source_file + page_number 去重
   ▼
Chainlit: _stream_text 逐 token 渲染
```

---

## 10. 关键设计决策与取舍

| 决策 | 理由 | 代价 |
|---|---|---|
| **LangGraph 而非 ReAct** | 状态图 + 条件边 + 循环原生支持,显式可控、可调试 | 学习曲线略陡 |
| **BM25 + Dense 双路 + RRF** | 互补(精确术语 vs 语义),无训练、可解释 | 召回需双索引 |
| **Cross-Encoder 重排可选** | 精度显著提升,但可禁用(None)以省 GPU | 重排有延迟 |
| **TypedDict 状态 + Pydantic Schema 分离** | LangGraph reducer 用 TypedDict,对外契约用 Pydantic | 两套定义 |
| **离线确定性模式** | 无 key 也能测全控制流 | 生成质量低(仅测试用) |
| **原子块保护** | 公式/表格完整,学术场景必需 | 偶有超大块(仅 warning) |
| **`where` 仅 dense** | BM25 无 metadata 过滤能力 | 过滤不作用于稀疏路 |
| **真实文献测试集** | 暴露合成数据测不到的 bug(代理字符) | 抓取受 arXiv 限流 |

---

## 11. 目录结构

```
RAG/
├── app.py                       # Chainlit 聊天前端
├── main.py                      # 环境自检入口
├── config/config.yaml           # 唯一参数源
├── data/
│   ├── pdfs/                    # 7 篇真实 arXiv PDF
│   └── arxiv_testset.json       # 元数据清单
├── src/                         # Python 包
│   ├── common/                  # config · logging · schemas
│   ├── ingestion/               # parser · splitter · embedder · vectorstore · ...
│   ├── retrieval/               # bm25 · rrf · reranker · hybrid
│   ├── agent/                   # llm_client · prompts · graph
│   └── evaluation/              # dataset · metrics · judge
├── tests/                       # 54 单元测试 + 评估管道
│   ├── data/                    # eval_queries*.json
│   └── evaluation/              # run_eval*.py
├── scripts/                     # arXiv 抓取 / 演示 / 在线检查
├── Dockerfile / .dockerignore   # 容器化
└── requirements.txt
```

---

## 12. 扩展指南

| 想做什么 | 改哪里 |
|---|---|
| **换 LLM provider** | `.env`(`OPENAI_API_KEY`/`OPENAI_BASE_URL`)+ `config.yaml`(`agent.llm.model`),零代码 |
| **启用真实 dense 检索** | 构建带 `VectorIndex` 的 `HybridRetriever`(加载 `bge-large-zh`) |
| **启用交叉编码器重排** | 向 `HybridRetriever` 传 `CrossEncoderReranker`(`bge-reranker-large`) |
| **新增 Agent 节点** | `AgentNodes` 加方法 + `graph.add_node` + 接条件边 |
| **新增评估指标** | `metrics.py` 加纯函数 + `LLMJudge` 加方法 |
| **扩充语料** | `python scripts/fetch_arxiv_testset.py --total N` |
| **调分块/检索参数** | `config.yaml` 对应字段 |
| **接多模态** | parser 扩图表提取 → chunk;Agent 加视觉节点 |

---

<div align="center">

**4 层架构 · 6 节点状态机 · 双重防死循环 · 零幻觉生成**

</div>

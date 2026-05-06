# 固定收益产品投研问答系统

基于混合检索（BGE 向量 + BM25 关键词）、Cross-Encoder 重排序和时效性衰减的固定收益研报 RAG 问答系统。支持语义检索、精确数值校验与引用溯源。

## 背景

面向基金公司投研团队，构建一套债券市场研报问答系统，覆盖利率分析、信用评级、债券估值等场景。核心解决三个问题：
- 金融文档的精准检索（混合检索 + 重排序）
- 过期研报的时效性控制（发布日期加权衰减）
- LLM 输出的数值准确性（数值格式校验 + 强制引用来源）

## 技术栈

`Python` `FastAPI` `Qdrant` `BGE` `BM25` `Cross-Encoder` `DeepSeek API`

## 核心特性

- **混合检索**：BGE 稠密向量 + BM25 稀疏关键词 → RRF 融合 → Cross-Encoder 重排序 → 时效性衰减
- **业务驱动 Chunking**：按研报章节标题切分，保留完整业务逻辑单元
- **金融数值校验**：正则校验利率、日期、金额格式，格式不符触发 LLM 重试
- **引用溯源**：System Prompt 强制标注引用来源，拒绝编造
- **消融实验**：4 步逐步消融，量化每个组件的 Recall@10 增益

## 快速开始

### 环境要求

- Python 3.10+
- Qdrant（Docker 或本地运行）
- DeepSeek API Key

### 安装依赖

```bash
pip install -r requirements.txt


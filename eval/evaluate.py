from __future__ import annotations

import sys
from pathlib import Path

# 支持从仓库根目录执行: python tasks/task1_fixed_income_rag/eval/evaluate.py
ROOT_TASK = Path(__file__).resolve().parent.parent
BIG_PROJECT = ROOT_TASK.parent.parent
sys.path.insert(0, str(BIG_PROJECT))
sys.path.insert(0, str(ROOT_TASK))

import json
import asyncio
import re
from typing import List, Dict, Optional

from src.retriever import UnifiedRetriever
from src.llm_generator import generate_answer

# test_set.json 中 [cite: k] 对应 data/bond_reports 文件名（须与入库 payload.source 一致）
CITE_INDEX_TO_SOURCE_FILE = {
    "1": "固收双周报 2026年2月11日.md",
    "2": "固投双周报 2026年3月11日.md",
    "3": "固投双周报 2026年3月25日.md",
    "4": "固投双周报 2026年4月8日.md",
    "5": "固投双周报 2026年4月22日.md",
    "6": "固投双周报 2026年2月25日.md",
    "7": "2025 年债券市场发展报告.md",
}


def _normalize_source_name(name: str) -> str:
    """合并重复空格等，避免文件名轻微不一致导致 Recall 恒为 0。"""
    return " ".join((name or "").split())


def _date_key_from_source_filename(name: str) -> Optional[str]:
    """从研报文件名解析日期，用于 Relaxed Recall（同日不同文件名视为弱等价）。"""
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", name)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{y:04d}-{mo:02d}-{d:02d}"


def relevant_sources_for_item(item: Dict) -> List[str]:
    """
    优先使用条目中的 relevant_docs（文件名列表）；
    否则解析 source 字段里的 [cite: n] / [cite: n, m]，映射为研报文件名。
    """
    explicit = item.get("relevant_docs")
    if explicit:
        return [str(x) for x in explicit if x]
    raw = (item.get("source") or "").strip()
    if not raw or raw.upper() == "N/A":
        return []
    m = re.search(r"cite:\s*([\d\s,]+)", raw, re.IGNORECASE)
    if not m:
        return []
    parts = re.split(r"[\s,]+", m.group(1).strip())
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        fn = CITE_INDEX_TO_SOURCE_FILE.get(p)
        if fn:
            out.append(fn)
    return out


class Evaluator:
    def __init__(self, test_set_path: Optional[str] = None):
        path = Path(test_set_path) if test_set_path else ROOT_TASK / "eval" / "test_set.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.test_set = data.get("sample_questions", [])
        self.retriever = UnifiedRetriever()
        self.results = []

    async def run_ablation(self):
        """运行四步消融实验"""
        print("=" * 60)
        print("消融实验开始")
        print("=" * 60)

        ablation_steps = [
            ("step1_vector_only", "基础向量检索"),
            ("step2_add_bm25", "+BM25 混合"),
            ("step3_add_rerank", "+Rerank 重排序"),
            ("step4_add_time_decay", "+时效性衰减"),
        ]

        all_results = {name: [] for name, _ in ablation_steps}

        for i, item in enumerate(self.test_set):
            query = item["question"]
            relevant_sources = relevant_sources_for_item(item)
            ground_truth = item.get("ground_truth", "")

            print(f"\n[{i+1}/{len(self.test_set)}] Query: {query[:50]}...")

            for step_name, step_label in ablation_steps:
                docs = await self._retrieve_by_step(step_name, query)
                recall = self._compute_recall(docs, relevant_sources)
                recall_relaxed = self._compute_recall_relaxed(docs, relevant_sources)

                all_results[step_name].append({
                    "query_id": item.get("id", i),
                    "query": query,
                    "recall": recall,
                    "recall_relaxed": recall_relaxed,
                    "retrieved_count": len(docs),
                })

                print(
                    f"  {step_label}: Recall@10 strict={recall:.3f}, "
                    f"relaxed={recall_relaxed:.3f}"
                )

        # 打印汇总
        print("\n" + "=" * 60)
        print("消融实验汇总")
        print("=" * 60)
        for step_name, step_label in ablation_steps:
            n = len(all_results[step_name])
            avg_recall = sum(r["recall"] for r in all_results[step_name]) / n
            avg_relaxed = sum(r["recall_relaxed"] for r in all_results[step_name]) / n
            print(
                f"{step_label}: 平均 Recall@10 strict={avg_recall:.3f}, "
                f"relaxed={avg_relaxed:.3f}"
            )

        return all_results

    async def _retrieve_by_step(self, step_name: str, query: str) -> List[Dict]:
        """根据消融步骤执行不同的检索策略"""
        if step_name == "step1_vector_only":
            return await self.retriever.step1_vector_only(query)
        elif step_name == "step2_add_bm25":
            return await self.retriever.step2_add_bm25(query)
        elif step_name == "step3_add_rerank":
            return await self.retriever.step3_add_rerank(query)
        elif step_name == "step4_add_time_decay":
            return await self.retriever.step4_add_time_decay(query)
        else:
            return await self.retriever.search(query)

    def _compute_recall(self, docs: List[Dict], relevant_sources: List[str]) -> float:
        """Strict Recall@10：标注文件名与检索结果 source 完全一致。"""
        if not relevant_sources:
            return 0.0
        retrieved_names = set()
        for d in docs[:10]:
            meta = d.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            src = meta.get("source")
            if src:
                retrieved_names.add(_normalize_source_name(src))
        want = {_normalize_source_name(s) for s in relevant_sources}
        recalled = sum(1 for s in want if s in retrieved_names)
        return recalled / len(want)

    def _compute_recall_relaxed(self, docs: List[Dict], relevant_sources: List[str]) -> float:
        """
        Relaxed Recall@10：Strict 命中计 1；否则若检索结果中存在与标注文件「同日」的研报文件名，
        视为语义弱等价（同日不同刊名/栏目）。
        """
        if not relevant_sources:
            return 0.0
        retrieved_list: List[str] = []
        for d in docs[:10]:
            meta = d.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            src = meta.get("source")
            if src:
                retrieved_list.append(_normalize_source_name(src))

        want_list = [_normalize_source_name(s) for s in relevant_sources]
        hits = 0
        for w in want_list:
            if self._relaxed_source_hit(w, retrieved_list):
                hits += 1
        return hits / len(want_list)

    @staticmethod
    def _relaxed_source_hit(wanted_norm: str, retrieved_norms: List[str]) -> bool:
        if wanted_norm in retrieved_norms:
            return True
        dw = _date_key_from_source_filename(wanted_norm)
        if not dw:
            return False
        for r in retrieved_norms:
            dr = _date_key_from_source_filename(r)
            if dr and dr == dw:
                return True
        return False

    async def evaluate_end_to_end(self):
        """端到端评估：LLM 生成 + 引用准确率 + 数值校验"""
        print("\n" + "=" * 60)
        print("端到端评估（LLM 生成）")
        print("=" * 60)

        results = []
        citation_correct = 0
        total_retry = 0
        total_attempts = 0

        for i, item in enumerate(self.test_set):
            query = item["question"]
            ground_truth = item.get("ground_truth", "")

            print(f"\n[{i+1}/{len(self.test_set)}] Query: {query[:50]}...")

            docs = await self.retriever.step4_add_time_decay(query)
            answer, citations, num_retry = await generate_answer(query, docs)

            # 人工标注（这里用简化判断，实际需要人工逐条检查）
            citations_correct = self._check_citations_real(citations, docs)
            if citations_correct:
                citation_correct += 1

            total_retry += num_retry
            total_attempts += 1

            # 端到端准确率（基于 ground_truth 关键词匹配，实际需要人工标注）
            accuracy = self._judge_accuracy(answer, ground_truth)

            results.append({
                "query_id": item.get("id", i),
                "query": query,
                "answer": answer[:200],
                "citations": citations,
                "citations_correct": citations_correct,
                "num_retry": num_retry,
                "accuracy": accuracy,
            })

        # 汇总
        print("\n" + "=" * 60)
        print("端到端评估汇总")
        print("=" * 60)
        print(f"引用准确率: {citation_correct}/{total_attempts} = {citation_correct/total_attempts:.1%}")
        print(f"数值校验触发率: {total_retry} 次重试 / {total_attempts} 条查询")
        print(f"数值校验触发比例: {total_retry/total_attempts:.2f} 次/条")

        accurate_count = sum(1 for r in results if r["accuracy"] == "完全正确")
        partial_count = sum(1 for r in results if r["accuracy"] == "部分正确")
        wrong_count = sum(1 for r in results if r["accuracy"] == "错误")
        print(f"端到端准确率: 完全正确={accurate_count}, 部分正确={partial_count}, 错误={wrong_count}")

        return results

    def _check_citations_real(self, citations: List[str], docs: List[Dict]) -> bool:
        """检查 LLM 引用的来源是否真实存在于检索到的文档中"""
        # 提取检索到的文档 ID
        retrieved_ids = set()
        for doc in docs:
            source = doc.get("metadata", {}).get("source", doc.get("doc_id", ""))
            retrieved_ids.add(source)

        # 检查每个引用是否都存在于检索结果中
        for citation in citations:
            # citation 格式: "[1] source (date)"
            source_part = citation.split("] ")[-1].split(" (")[0] if "] " in citation else citation
            if source_part not in retrieved_ids:
                return False
        return len(citations) > 0

    def _judge_accuracy(self, answer: str, ground_truth: str) -> str:
        """判断端到端准确率（简化版，实际需要人工标注）"""
        if not ground_truth:
            return "未标注"

        # 简单的关键词匹配（实际评测需要人工逐条判断）
        gt_keywords = set(ground_truth.lower().split())
        answer_keywords = set(answer.lower().split())
        overlap = len(gt_keywords & answer_keywords)

        if overlap >= len(gt_keywords) * 0.8:
            return "完全正确"
        elif overlap >= len(gt_keywords) * 0.3:
            return "部分正确"
        else:
            return "错误"

    def export_results(self, ablation_results, e2e_results, output_path="eval/eval_results.json"):
        """导出评估结果为 JSON"""
        out = Path(output_path)
        if not out.is_absolute():
            out = ROOT_TASK / out
        def _avg(step: str, key: str) -> float:
            rows = ablation_results.get(step, [])
            return sum(r[key] for r in rows) / len(rows) if rows else 0.0

        output = {
            "ablation": {
                step: {
                    "avg_recall_strict": _avg(step, "recall"),
                    "avg_recall_relaxed": _avg(step, "recall_relaxed"),
                    "details": results
                }
                for step, results in ablation_results.items()
            },
            "end_to_end": e2e_results,
            "summary": {
                "vector_only_recall_strict": _avg("step1_vector_only", "recall"),
                "vector_only_recall_relaxed": _avg("step1_vector_only", "recall_relaxed"),
                "hybrid_recall_strict": _avg("step2_add_bm25", "recall"),
                "hybrid_recall_relaxed": _avg("step2_add_bm25", "recall_relaxed"),
                "rerank_recall_strict": _avg("step3_add_rerank", "recall"),
                "rerank_recall_relaxed": _avg("step3_add_rerank", "recall_relaxed"),
                "time_decay_recall_strict": _avg("step4_add_time_decay", "recall"),
                "time_decay_recall_relaxed": _avg("step4_add_time_decay", "recall_relaxed"),
                "citation_accuracy": f"{sum(1 for r in e2e_results if r['citations_correct'])}/{len(e2e_results)}",
                "avg_retry_per_query": f"{sum(r['num_retry'] for r in e2e_results) / len(e2e_results):.2f}" if e2e_results else "0",
            }
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到 {out}")


async def main():
    evaluator = Evaluator()

    # 步骤 1：消融实验
    ablation_results = await evaluator.run_ablation()

    # 步骤 2：端到端评估
    llm_output_path = ROOT_TASK / "eval" / "llm_outputs.txt"
    llm_output_path.parent.mkdir(parents=True, exist_ok=True)
    e2e_results = []

    with open(llm_output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(evaluator.test_set):
            query = item["question"]
            ground_truth = item.get("ground_truth", "")
            
            # 检索 + 生成
            docs = await evaluator.retriever.step4_add_time_decay(query)
            answer, citations, num_retry = await generate_answer(query, docs)
            citations_correct = evaluator._check_citations_real(citations, docs)
            accuracy = evaluator._judge_accuracy(answer, ground_truth)
            e2e_results.append({
                "query_id": item.get("id", i),
                "query": query,
                "answer": answer[:200],
                "citations": citations,
                "citations_correct": citations_correct,
                "num_retry": num_retry,
                "accuracy": accuracy,
            })
            
            # 写入文件
            f.write(f"[{i+1}] {query}\n")
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Answer: {answer}\n")
            f.write(f"引用: {citations}\n")
            f.write(f"重试次数: {num_retry}\n")
            f.write("-" * 60 + "\n\n")
            
            print(f"[{i+1}/{len(evaluator.test_set)}] {query[:40]}... 已保存")

    print(f"\nLLM 输出已保存到 {llm_output_path}，请打开文件进行人工标注。")

    # 步骤 3：导出结果
    evaluator.export_results(ablation_results, e2e_results)


if __name__ == "__main__":
    asyncio.run(main())

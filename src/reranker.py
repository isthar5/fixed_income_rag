"""将 UnifiedRetriever 的 List[Dict] 与 app.retrieval.reranker 的 Tuple 格式对接。"""
from __future__ import annotations

from typing import Any, Dict, List

from app.retrieval.reranker import Reranker as _BaseReranker


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self._inner = _BaseReranker(model_name=model_name, device=device)

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return documents
        tuples = [(str(d.get("doc_id", "")), d) for d in documents]
        ranked = self._inner.rerank(query, tuples)
        out: List[Dict[str, Any]] = []
        for doc_id, info in ranked:
            row = dict(info)
            row["doc_id"] = doc_id
            out.append(row)
        return out

import asyncio
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector
import numpy as np
import hashlib

from app.config.settings import settings
from app.retrieval.embedder import embed
from .reranker import Reranker

logger = logging.getLogger("retriever.unified")

RRF_K = 60  # Reciprocal Rank Fusion 平滑常数
_recent_query_cache: Dict[str, List[Dict]] = {}


class SearchResult:
    def __init__(self, doc_id: str, score: float, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata

    def dict(self):
        return {"doc_id": self.doc_id, "score": self.score, "metadata": self.metadata}


class UnifiedRetriever:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = settings.COLLECTION_NAME
        self.reranker = Reranker()
        self.top_k_default = 5
        self.semaphore = asyncio.Semaphore(4)

        # 时效性衰减系数（较小，避免过度压制旧文档）
        self.time_decay_lambda = 0.01
        self.now = datetime.utcnow()

    # ================== 基础检索（单路） ==================
    async def _dense_search(self, query_vector, limit: int) -> List[Any]:
        resp = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_vector,
            using="dense_vector",
            limit=limit,
            with_payload=True,
        )
        return resp.points if resp else []

    async def _sparse_search(self, sparse_query, limit: int) -> List[Any]:
        if isinstance(sparse_query, dict):
            sq = SparseVector(**sparse_query)
        else:
            sq = sparse_query
        resp = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=sq,
            using="bm25",
            limit=limit,
            with_payload=True,
        )
        return resp.points if resp else []

    # ================== 统一入口（完整管道） ==================
    async def search(self, query: str, top_k: int = None, rerank: bool = True) -> List[Dict]:
        top_k = top_k or self.top_k_default

        q_hash = hashlib.md5(query.encode()).hexdigest()
        if q_hash in _recent_query_cache:
            logger.info(f"[CACHE HIT] query: {query}")
            return _recent_query_cache[q_hash][:top_k]

        try:
            fuse_limit = max(60, top_k * 6) if rerank else max(20, top_k * 3)
            fused = await self.hybrid_search_logic(query, limit=fuse_limit)
            if rerank and fused:
                fused = self._rerank_document_level(query, fused, top_k)

            _recent_query_cache[q_hash] = fused
            return fused[:top_k]
        except Exception as e:
            logger.error(f"[Retriever Error] query: {query} -> {e}")
            return []

    # ================== 混合检索逻辑 ==================
    async def hybrid_search_logic(
        self, query: str, limit: int = 20, method: str = "rrf", weights: Tuple[float, float] = None
    ) -> List[Dict]:
        async with self.semaphore:
            dense_vec, sparse_vec = await embed(query)

            dense_pts, sparse_pts = await asyncio.gather(
                self._dense_search(dense_vec, limit),
                self._sparse_search(sparse_vec, limit),
                return_exceptions=True,
            )
            if isinstance(dense_pts, Exception):
                logger.error(f"dense search error: {dense_pts}")
                dense_pts = []
            if isinstance(sparse_pts, Exception):
                logger.error(f"sparse search error: {sparse_pts}")
                sparse_pts = []

            return self.hybrid_fusion(dense_pts, sparse_pts, method, weights, query)

    # ================== 融合逻辑 ==================
    def hybrid_fusion(
        self,
        dense_hits,
        sparse_hits,
        method="rrf",
        weights=None,
        query=None
    ) -> List[Dict]:
        if method == "rrf":
            return self._rrf_fuse(dense_hits, sparse_hits, weights, query)
        else:
            return self._weighted_fuse(dense_hits, sparse_hits, weights, query)

    def _compute_time_decay(self, metadata: Dict[str, Any]) -> float:
        pub_date = metadata.get("publish_date")
        if not pub_date:
            return 1.0
        try:
            if isinstance(pub_date, str):
                pub_date = datetime.fromisoformat(pub_date)
            days = (self.now - pub_date).days
            return float(np.exp(-self.time_decay_lambda * days))
        except Exception:
            return 1.0

    def _rrf_fuse(self, dense_hits, sparse_hits, weights, query) -> List[Dict]:
        dw, sw = weights if weights else (0.5, 0.5)
        scores: Dict[Any, float] = {}
        payloads: Dict[Any, Dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            pid = getattr(hit, "id", None)
            if pid is None:
                continue
            scores[pid] = scores.get(pid, 0.0) + dw / (RRF_K + rank)
            if pid not in payloads:
                payloads[pid] = getattr(hit, "payload", {}) or {}

        for rank, hit in enumerate(sparse_hits, start=1):
            pid = getattr(hit, "id", None)
            if pid is None:
                continue
            scores[pid] = scores.get(pid, 0.0) + sw / (RRF_K + rank)
            if pid not in payloads:
                payloads[pid] = getattr(hit, "payload", {}) or {}

        result = []
        for pid, sc in scores.items():
            pl = payloads.get(pid, {})
            decay = self._compute_time_decay(pl if isinstance(pl, dict) else {})
            final_score = sc * decay
            result.append((pid, final_score, pl))

        result.sort(key=lambda x: x[1], reverse=True)
        return [{"doc_id": r[0], "score": r[1], "metadata": r[2]} for r in result]

    def _weighted_fuse(self, dense_hits, sparse_hits, weights, query) -> List[Dict]:
        if weights is None:
            weights = self.dynamic_weights(query)
        dw, sw = weights
        scores: Dict[Any, float] = {}
        payloads: Dict[Any, Dict] = {}

        for hit in dense_hits:
            pid = getattr(hit, "id", None)
            if pid is None:
                continue
            sc = float(getattr(hit, "score", 0.0) or 0.0) * dw
            scores[pid] = scores.get(pid, 0.0) + sc
            if pid not in payloads:
                payloads[pid] = getattr(hit, "payload", {}) or {}

        for hit in sparse_hits:
            pid = getattr(hit, "id", None)
            if pid is None:
                continue
            sc = float(getattr(hit, "score", 0.0) or 0.0) * sw
            scores[pid] = scores.get(pid, 0.0) + sc
            if pid not in payloads:
                payloads[pid] = getattr(hit, "payload", {}) or {}

        result = []
        for pid, sc in scores.items():
            pl = payloads.get(pid, {})
            decay = self._compute_time_decay(pl if isinstance(pl, dict) else {})
            final_score = sc * decay
            result.append((pid, final_score, pl))

        result.sort(key=lambda x: x[1], reverse=True)
        return [{"doc_id": r[0], "score": r[1], "metadata": r[2]} for r in result]

    @staticmethod
    def dynamic_weights(query: str) -> Tuple[float, float]:
        if len(query) < 10:
            return 0.3, 0.7
        return 0.7, 0.3

    # ================== 文档级 Rerank ==================
    def _document_source_key(self, d: Dict[str, Any]) -> str:
        meta = d.get("metadata")
        if isinstance(meta, dict):
            src = meta.get("source")
            if src:
                return str(src).strip()
        return str(d.get("doc_id", ""))

    def _rerank_document_level(self, query: str, fused_chunks: List[Dict], k: int) -> List[Dict]:
        """文档级聚合：每个 source 保留最高分的一个 chunk，再精排"""
        doc_best: Dict[str, Dict] = {}
        for d in fused_chunks:
            key = self._document_source_key(d)
            if not key:
                continue
            prev = doc_best.get(key)
            if prev is None or float(d.get("score", 0.0)) > float(prev.get("score", 0.0)):
                doc_best[key] = d

        doc_list = list(doc_best.values())
        if not doc_list:
            return []
        reranked = self.reranker.rerank(query, doc_list)
        return reranked[:k]

    # ================== 消融实验四个步骤 ==================
    async def step1_vector_only(self, query, k=10):
        dense_vec, _ = await embed(query)
        points = await self._dense_search(dense_vec, k)
        out = []
        for p in points:
            pl = getattr(p, "payload", None) or {}
            if not isinstance(pl, dict):
                pl = {}
            out.append({
                "doc_id": str(getattr(p, "id", "")),
                "score": float(getattr(p, "score", 0.0) or 0.0),
                "metadata": pl,
            })
        return out

    async def step2_add_bm25(self, query, k=10):
        fused = await self.hybrid_search_logic(query, limit=k)
        return fused[:k]

    async def step3_add_rerank(self, query, k=10):
        fused = await self.hybrid_search_logic(query, limit=k * 3)
        return self._rerank_document_level(query, fused, k)

    async def step4_add_time_decay(self, query, k=10):
        return await self.search(query, top_k=k, rerank=True)

    def compute_recall(self, docs, relevant_doc_ids):
        if not relevant_doc_ids:
            return 0.0
        retrieved_ids = [d["doc_id"] for d in docs[:10]]
        recalled = sum(1 for rid in relevant_doc_ids if rid in retrieved_ids)
        return recalled / len(relevant_doc_ids)
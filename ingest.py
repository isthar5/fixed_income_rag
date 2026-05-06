"""
将 data/bond_reports 下 .md / .txt 切块后写入 Qdrant（与大项目 app/ingestion 向量配置一致）。

运行（在大项目根目录已配置 PYTHONPATH 时）:
  python tasks/task1_fixed_income_rag/ingest.py

或在本子目录:
  python ingest.py
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

TASK_ROOT = Path(__file__).resolve().parent
BIG_PROJECT = TASK_ROOT.parent.parent
sys.path.insert(0, str(BIG_PROJECT))
sys.path.insert(0, str(TASK_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config.settings import settings
from app.retrieval.embedder import embed
from src.chunker import BondReportChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOND_REPORTS = TASK_ROOT / "data" / "bond_reports"
BATCH_SIZE = 32


def guess_date_from_name(stem: str) -> str:
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", stem)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m2 = re.search(r"(20\d{2})\s*年", stem)
    if m2:
        return f"{m2.group(1)}-01-01"
    return "1970-01-01"


def stable_point_id(text: str, source: str, idx: int) -> int:
    u = f"{source}_{idx}_{text[:100]}"
    return int(hashlib.md5(u.encode("utf-8")).hexdigest()[:16], 16)


class BondReportIngestor:
    def __init__(self) -> None:
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.COLLECTION_NAME
        self.chunker = BondReportChunker(token_limit=500)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            logger.info("集合已存在: %s", self.collection_name)
            return
        logger.info("创建集合 %s", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": qmodels.VectorParams(
                    size=512,
                    distance=qmodels.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": qmodels.SparseVectorParams(index=qmodels.SparseIndexParams())
            },
        )

    async def ingest_file(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 20:
            return []
        stem = path.stem
        pub = guess_date_from_name(stem)
        source = path.name
        chunks = await self.chunker.chunk_report(
            text, publish_date=pub, report_type="bond", company=""
        )
        points: List[Dict[str, Any]] = []
        for idx, ch in enumerate(chunks):
            body = ch.get("text") or ""
            if len(body.strip()) < 10:
                continue
            dense, sparse = await embed(body)
            pid = stable_point_id(body, source, idx)
            payload = {
                "text": body[:3000],
                "source": source,
                "publish_date": ch.get("publish_date", pub),
                "chunk_idx": idx,
                "report_type": ch.get("report_type", "bond"),
                "company": ch.get("company", ""),
            }
            points.append(
                {
                    "id": pid,
                    "vector": {"dense_vector": dense, "bm25": sparse},
                    "payload": payload,
                }
            )
        return points

    def upsert_batch(self, points: List[Dict[str, Any]]) -> None:
        if not points:
            return
        qdrant_points = [
            qmodels.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points,
        )

    async def ingest_bond_reports(self) -> int:
        self.ensure_collection()
        files = sorted(p for p in BOND_REPORTS.iterdir() if p.suffix.lower() in {".md", ".txt"})
        logger.info("共 %s 个研报文件", len(files))
        buf: List[Dict[str, Any]] = []
        total = 0
        for fp in files:
            pts = await self.ingest_file(fp)
            buf.extend(pts)
            total += len(pts)
            while len(buf) >= BATCH_SIZE:
                batch = buf[:BATCH_SIZE]
                buf = buf[BATCH_SIZE:]
                self.upsert_batch(batch)
            logger.info("  %s -> %s chunks", fp.name, len(pts))
        if buf:
            self.upsert_batch(buf)
        logger.info("入库完成，累计 points %s", total)
        return total


async def main() -> None:
    ingestor = BondReportIngestor()
    await ingestor.ingest_bond_reports()


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - 可选依赖
    tiktoken = None  # type: ignore


class BondReportChunker:
    def __init__(self, encoding_name: str = "cl100k_base", token_limit: int = 500):
        self.token_limit = token_limit
        self.encoding: Optional[Any] = None
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                self.encoding = None

    def _count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        return max(1, int(len(text) / 1.5))

    async def chunk_report(
        self,
        text: str,
        publish_date: str,
        report_type: str,
        company: str,
        overlap_tokens: int = 50,
    ) -> List[Dict]:
        paragraphs = self._split_by_headers(text)
        chunks: List[Dict] = []
        current_lines: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            t = self._count_tokens(para)
            if current_lines and current_tokens + t > self.token_limit:
                chunks.append(
                    self._build_chunk(current_lines, publish_date, report_type, company)
                )
                overlap_text = self._get_overlap_text(current_lines, overlap_tokens)
                current_lines = [overlap_text] if overlap_text else []
                current_tokens = (
                    self._count_tokens(" ".join(current_lines)) if current_lines else 0
                )
            current_lines.append(para)
            current_tokens += t

        if current_lines:
            chunks.append(
                self._build_chunk(current_lines, publish_date, report_type, company)
            )
        return chunks

    def _split_by_headers(self, text: str) -> List[str]:
        pattern = r"(?:\n|^)(\d{1,2}\.?[\d]*\s*[^\n]{2,20})\n"
        headers = list(re.finditer(pattern, text))
        paragraphs: List[str] = []

        if not headers:
            return [p.strip() for p in text.split("\n") if p.strip()]

        last_idx = 0
        for h in headers:
            start = h.start()
            if start > last_idx:
                para = text[last_idx:start].strip()
                if para:
                    paragraphs.append(para)
            header_text = text[h.start() : h.end()].strip()
            paragraphs.append(header_text)
            last_idx = h.end()
        tail = text[last_idx:].strip()
        if tail:
            paragraphs.append(tail)
        return paragraphs

    def _get_overlap_text(self, chunk_lines: List[str], overlap_tokens: int) -> Optional[str]:
        if not chunk_lines:
            return None
        text = "\n".join(chunk_lines)
        if not self.encoding:
            approx_chars = int(overlap_tokens * 1.5)
            return text[-approx_chars:] if len(text) > approx_chars else text
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        return self.encoding.decode(tokens[-overlap_tokens:])

    def _build_chunk(
        self,
        chunk_lines: List[str],
        publish_date: str,
        report_type: str,
        company: str,
    ) -> Dict:
        body = "\n".join(chunk_lines)
        return {
            "chunk_id": str(uuid.uuid4()),
            "text": body,
            "publish_date": publish_date,
            "report_type": report_type,
            "company": company,
        }

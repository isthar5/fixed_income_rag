import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple

from openai import AsyncOpenAI
from app.config.settings import settings

logger = logging.getLogger("llm.financial_answer")

llm_client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL
)

FINANCIAL_NUMBER_PATTERNS = {
    "rate": re.compile(r'\b\d+\.?\d*\s*%'),
    "date": re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    "amount": re.compile(r'\b\d+\.?\d*\s*(亿|万|千)?元\b'),
}


def temporal_weight(publish_date: str, lambda_decay: float = 0.01) -> float:
    """时效性衰减函数：越新文档权重越高"""
    try:
        doc_date = datetime.strptime(publish_date, "%Y-%m-%d")
        age_days = (datetime.now() - doc_date).days
        return 1.0 * (2.71828 ** (-lambda_decay * age_days))
    except Exception:
        return 1.0


def validate_financial_numbers(text: str) -> bool:
    """校验 LLM 输出中的金融数值格式和引用来源"""
    has_number = any(p.search(text) for p in FINANCIAL_NUMBER_PATTERNS.values())
    has_citation = bool(re.search(r'\[\d+\]', text))
    has_decline = any(kw in text for kw in ["未提及", "找不到", "无相关数据"])
    return has_decline or (has_number and has_citation)


def extract_citations(docs: List[Dict], top_n: int = 5) -> List[str]:
    """提取引用来源列表"""
    citations = []
    for i, doc in enumerate(docs[:top_n], 1):
        source = doc.get("metadata", {}).get("source", doc.get("doc_id", "未知"))
        date = doc.get("metadata", {}).get("publish_date", "未知")
        citations.append(f"[{i}] {source} ({date})")
    return citations


class LLMGenerator:
    async def generate_answer(
        self,
        query: str,
        docs: List[Dict],
        top_n_docs: int = 5,
        max_retries: int = 3
    ) -> Tuple[str, List[str], int]:
        """
        Args:
            query: 用户提问
            docs: 检索到的文档列表
            top_n_docs: 使用前 N 条文档构建上下文
            max_retries: 数值校验失败时最大重试次数
        Returns:
            answer: LLM 输出
            citations: 引用来源列表
            num_retry: 数值校验触发次数
        """
        num_retry = 0

        docs_sorted = sorted(
            docs[:top_n_docs],
            key=lambda d: temporal_weight(d.get("metadata", {}).get("publish_date", "")),
            reverse=True
        )

        context_lines = []
        for i, doc in enumerate(docs_sorted, 1):
            md = doc.get("metadata", {})
            text = md.get("text", doc.get("text", ""))[:600]
            source = md.get("source", doc.get("doc_id", "未知"))
            date = md.get("publish_date", "未知")
            context_lines.append(f"[{i}] 来源：{source} ({date})\n{text}")

        context_str = "\n---\n".join(context_lines)

        system_prompt = """你是固定收益产品投研分析师。

重要规则：
1. 只引用检索到的研报内容，不可补充训练知识
2. 每个结论必须标注引用来源，例如 [1]、[2]
3. 未找到信息必须明确标注 "未提及"
4. 所有数值型数据（利率、金额、日期）必须标注在引用之后
5. 禁止编造任何数值"""

        user_prompt = f"""【用户问题】：{query}

【研报上下文】：
{context_str}

请给出专业分析（必须标注引用来源）："""

        answer = ""

        for attempt in range(max_retries):
            try:
                resp = await llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                answer = resp.choices[0].message.content

                if validate_financial_numbers(answer):
                    break
                else:
                    num_retry += 1
                    user_prompt += "\n\n【注意】上次回答中有数值格式错误，请确保利率、金额、日期格式正确。"
            except Exception as e:
                num_retry += 1
                answer = f"LLM 调用失败: {e}"
                logger.error(f"LLM error on attempt {attempt+1}: {e}")

        citations = extract_citations(docs_sorted, top_n=top_n_docs)
        return answer, citations, num_retry


async def generate_answer(
    query: str,
    docs: List[Dict],
    top_n_docs: int = 5,
    max_retries: int = 3,
) -> Tuple[str, List[str], int]:
    """供 eval 等脚本使用的模块级入口。"""
    gen = LLMGenerator()
    return await gen.generate_answer(query, docs, top_n_docs=top_n_docs, max_retries=max_retries)
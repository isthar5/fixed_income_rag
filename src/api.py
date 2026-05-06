from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from src.retriever import UnifiedRetriever
from src.llm_generator import LLMGenerator

app = FastAPI(title="Bond QA System")

retriever = UnifiedRetriever()
llm = LLMGenerator()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    top_docs: List[Dict]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    top_docs = await retriever.search(req.query, top_k=5, rerank=True)
    answer, citations, _ = await llm.generate_answer(req.query, top_docs)
    return QueryResponse(answer=answer, citations=citations, top_docs=top_docs)
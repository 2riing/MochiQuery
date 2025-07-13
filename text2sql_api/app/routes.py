# app/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from text2sql_api.app.milvus.rag_sql import generate_sql

router = APIRouter()

class QueryInput(BaseModel):
    question: str

@router.post("/text-to-sql")
async def text_to_sql(input_data: QueryInput):
    result = await generate_sql(input_data.question)
    return {"sql": result}
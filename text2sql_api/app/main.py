# app/main.py
from fastapi import FastAPI
# from app.routes import router
from text2sql_api.app.routes import router
import os

app = FastAPI(title="Text-to-SQL API with GPT, Milvus, Langserve")
app.include_router(router)

print(os.getenv("OPENAI_API_KEY"))
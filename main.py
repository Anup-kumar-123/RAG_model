import os
import tempfile
import shutil
from typing import List, Optional, Annotated
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from tavily import TavilyClient

from config import SIMILARITY_THRESHOLD, GEMINI_API_KEY, TAVILY_API_KEY
from ingestion import process_directory, extract_zip, parse_single_file
from vector_store import chunk_documents, build_vector_store
from audio_engine import generate_speech_stream

app = FastAPI(title="Gemini Multi-Doc RAG Backend")

vector_db: Optional[Chroma] = None

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


class QueryRequest(BaseModel):
    query: str
    allow_web_search: bool = False


@app.post("/upload")
async def upload_documents(files: Annotated[List[UploadFile], File(...)]):
    global vector_db
    all_docs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            if file.filename.endswith(".zip"):
                extract_dir = os.path.join(temp_dir, "unzipped_" + file.filename)
                extract_zip(file_path, extract_dir)
                docs, _ = process_directory(extract_dir)
                all_docs.extend(docs)
            else:
                all_docs.extend(parse_single_file(file_path, file.filename))

        if not all_docs:
            raise HTTPException(status_code=400, detail="No readable content found.")

        chunked = chunk_documents(all_docs)
        vector_db = build_vector_store(chunked)

    return {"status": "SUCCESS", "chunks_indexed": len(chunked)}


@app.post("/chat")
async def chat(req: QueryRequest):
    global vector_db
    if not vector_db:
        raise HTTPException(status_code=400, detail="Upload files first.")

    results = vector_db.similarity_search_with_relevance_scores(req.query, k=5)
    relevant_docs = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

    if not relevant_docs and not req.allow_web_search:
        return {
            "status": "REQUIRES_PERMISSION",
            "message": "I couldn't find relevant information in your uploaded files or code repository. Should I search the web globally?",
            "query": req.query
        }

    if relevant_docs:
        context = "\n\n".join([f"[{doc.metadata.get('source_path')}]: {doc.page_content}" for doc in relevant_docs])
        source = "documents"
    else:
        web_res = tavily_client.search(query=req.query, max_results=3)
        search_results = web_res.get("results", [])
        context = "\n\n".join([f"Web Result: {res.get('content', '')}" for res in search_results])
        source = "web_search"

    prompt = f"Answer the question using ONLY the provided context:\n\nContext:\n{context}\n\nQuestion: {req.query}"
    answer = llm.invoke(prompt).content

    return {"status": "SUCCESS", "answer": answer, "source": source}


@app.post("/tts")
async def tts(text: str = Form(...)):
    return StreamingResponse(generate_speech_stream(text), media_type="audio/mpeg")
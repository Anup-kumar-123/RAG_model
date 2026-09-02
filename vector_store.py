import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import GEMINI_API_KEY

LANG_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".java": Language.JAVA,
    ".html": Language.HTML
}

def chunk_documents(docs: List[Document]) -> List[Document]:
    chunked_docs = []
    default_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for doc in docs:
        path = doc.metadata.get("source_path", "")
        ext = os.path.splitext(path)[1].lower()

        if ext in LANG_MAP:
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=LANG_MAP[ext], chunk_size=1000, chunk_overlap=150
            )
            chunked_docs.extend(code_splitter.split_documents([doc]))
        else:
            chunked_docs.extend(default_splitter.split_documents([doc]))

    return chunked_docs

def build_vector_store(chunked_docs: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
    # Uses API instead of heavy local torch models
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=GEMINI_API_KEY
    )

    return Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

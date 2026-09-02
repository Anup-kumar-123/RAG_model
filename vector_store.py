import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
    # Switched to HuggingFace local embeddings (No Google API 404 errors)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
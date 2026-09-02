import os
import zipfile
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from config import EXCLUDED_DIRS


def build_folder_tree(dir_path: str) -> str:
    tree_str = "Project Directory Structure:\n"
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        level = root.replace(dir_path, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_str += f"{indent}{os.path.basename(root)}/\n"
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            tree_str += f"{sub_indent}{f}\n"
    return tree_str


def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            parts = file.split('/')
            if not any(excluded in parts for excluded in EXCLUDED_DIRS):
                zip_ref.extract(file, extract_to)


def parse_single_file(file_path: str, rel_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
        elif ext in [".txt", ".md", ".py", ".js", ".ts", ".cpp", ".java", ".html", ".css", ".json"]:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

        for doc in docs:
            doc.metadata["source_path"] = rel_path
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

    return docs


def process_directory(dir_path: str) -> Tuple[List[Document], str]:
    all_docs = []
    folder_tree = build_folder_tree(dir_path)

    tree_doc = Document(
        page_content=folder_tree,
        metadata={"source_path": "DIRECTORY_STRUCTURE.txt"}
    )
    all_docs.append(tree_doc)

    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, dir_path)
            docs = parse_single_file(full_path, rel_path)
            all_docs.extend(docs)

    return all_docs, folder_tree
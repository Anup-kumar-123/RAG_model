import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv",
    "dist", "build", ".idea", ".vscode", "coverage"
}

SIMILARITY_THRESHOLD = 0.1
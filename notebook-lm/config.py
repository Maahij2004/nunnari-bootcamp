import os
from dotenv import load_dotenv

load_dotenv()

# Model Settings
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Tool Settings
TAVILY_API_KEY = os.getenv("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")

# Paths
CHROMA_PATH = "storage/chroma_db"
NOTES_PATH = "storage/notes"
UPLOADS_PATH = "storage/uploads"

# Ensure directories exist
for path in [CHROMA_PATH, NOTES_PATH, UPLOADS_PATH]:
    os.makedirs(path, exist_ok=True)
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

# ==============================
# VALIDATE ENV
# ==============================
REQUIRED_ENV_VARS = [
    "CHROMA_QUERY_EXAMPLES",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_EMBEDDING_DEPLOYMENT_NAME"
]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"❌ Missing required environment variable: {var}")

# ==============================
# CONFIG
# ==============================
CHROMA_PATH = os.environ["CHROMA_QUERY_EXAMPLES"]

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_type="azure",
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    model_name=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
)

# ==============================
# CHROMA CLIENT
# ==============================
client = chromadb.PersistentClient(path=CHROMA_PATH)

# ==============================
# INGEST LOGIC
# ==============================
def ingest(collection_name: str, json_file: str):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"❌ JSON file not found: {json_file}")

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"❌ Invalid or empty JSON in {json_file}")

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    existing = collection.count()
    if existing > 0:
        print(f"ℹ️ Collection '{collection_name}' already has {existing} records — skipping ingest")
        return

    collection.add(
        ids=[f"{collection_name}_{i}" for i in range(len(data))],
        documents=[item["input"] for item in data],
        metadatas=[{"query": item["query"]} for item in data]
    )

    print(f"✅ Ingested {len(data)} records into '{collection_name}'")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("🚀 Starting Chroma example ingestion...")

    ingest("examples_generic", "sql_query_examples_generic.json")
    ingest("examples_usecase", "sql_query_examples_usecase.json")

    print("🎉 Ingestion complete")

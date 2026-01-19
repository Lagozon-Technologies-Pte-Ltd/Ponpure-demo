from openai import AzureOpenAI
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Azure OpenAI
# ==============================
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

# ==============================
# Chroma
# ==============================
chroma_client = chromadb.PersistentClient(
    path=os.environ["CHROMA_QUERY_EXAMPLES"]
)

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_type="azure",
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    model_name=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
)

# ==============================
# Collections (SAFE)
# ==============================
collections = {
    "generic": chroma_client.get_or_create_collection(
        name="examples_generic",
        embedding_function=embedding_function
    ),
    "usecase": chroma_client.get_or_create_collection(
        name="examples_usecase",
        embedding_function=embedding_function
    )
}

# ==============================
# Auto-ingest (WORKER SAFE)
# ==============================
def ensure_ingested(collection, json_file, prefix):
    if collection.count() > 0:
        return

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    collection.add(
        ids=[f"{prefix}_{i}" for i in range(len(data))],
        documents=[x["input"] for x in data],
        metadatas=[{"query": x["query"]} for x in data]
    )

# Run once per worker — safe because of count() check
ensure_ingested(collections["generic"], "sql_query_examples_generic.json", "generic")
ensure_ingested(collections["usecase"], "sql_query_examples_usecase.json", "usecase")

# ==============================
# QUERY FUNCTION
# ==============================
def get_examples(query: str, question_type: str):
    if question_type not in collections:
        raise ValueError("question_type must be 'generic' or 'usecase'")

    embedding = client.embeddings.create(
        input=[query],
        model=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
    ).data[0].embedding

    result = collections[question_type].query(
        query_embeddings=[embedding],
        n_results=2
    )

    return [
        {"input": doc, "query": meta["query"]}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]

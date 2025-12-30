import chromadb
import json
import os
from chromadb.utils import embedding_functions

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

client = chromadb.PersistentClient(path=CHROMA_PATH)


def ingest(collection_name, json_file):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    if collection.count() == 0:
        collection.add(
            ids=[f"{collection_name}_{i}" for i in range(len(data))],
            documents=[x["input"] for x in data],
            metadatas=[{"query": x["query"]} for x in data]
        )
        print(f"✅ Ingested {len(data)} records into {collection_name}")
    else:
        print(f"ℹ️ {collection_name} already exists — skipping")


# ------------------------------
# RUN ONCE
# ------------------------------
ingest("examples_generic", "sql_query_examples_generic.json")
ingest("examples_usecase", "sql_query_examples_usecase.json")

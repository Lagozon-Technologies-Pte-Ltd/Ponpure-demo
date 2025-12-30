from openai import AzureOpenAI
import os
from chromadb.utils import embedding_functions
import chromadb

# ==============================
# Azure OpenAI
# ==============================
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

# ==============================
# Chroma (READ ONLY)
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

collections = {
    "generic": chroma_client.get_collection(
        "examples_generic",
        embedding_function=embedding_function
    ),
    "usecase": chroma_client.get_collection(
        "examples_usecase",
        embedding_function=embedding_function
    )
}


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
        {"input": doc, "query": meta}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]

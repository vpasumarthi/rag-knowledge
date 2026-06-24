"""Shared embedding function for consistent model usage across ingest/query/MCP."""

from chromadb import EmbeddingFunction, Documents, Embeddings

MODEL_NAME = "BAAI/bge-base-en-v1.5"

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


class FastEmbedFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function using fastembed BGE-base-en-v1.5."""

    def __call__(self, input: Documents) -> Embeddings:
        model = _get_model()
        embeddings = list(model.embed(input))
        return [e.tolist() for e in embeddings]


def get_embedding_function() -> FastEmbedFunction:
    return FastEmbedFunction()

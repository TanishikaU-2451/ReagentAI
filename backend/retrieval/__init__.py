"""ReagentAI Retrieval Package -- Phase 2 RAG Knowledge Engine.

This package provides the Retrieval-Augmented Generation (RAG) pipeline
for ReagentAI, enabling the system to index research papers and other
documents into a vector store, then retrieve semantically relevant
passages to augment LLM prompts with grounded context.

Phase 4: GitHub repository search, ranking, and code extraction.
Phase 5: HuggingFace model hub retrieval.

Modules:
    embedding_model         : Sentence-transformer wrapper for text embedding.
    vector_store            : FAISS-backed vector storage and similarity search.
    rag_pipeline            : End-to-end chunking, indexing, and retrieval pipeline.
    github_search           : GitHub API repository search by paper keywords.
    repo_ranker             : Relevance-based ranking of retrieved repositories.
    code_extractor          : Extract key source files from GitHub repositories.
    huggingface_retrieval   : HuggingFace model hub search and retrieval.
"""

from backend.retrieval.embedding_model import EmbeddingModel
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.rag_pipeline import RAGPipeline
from backend.retrieval.github_search import GitHubSearcher
from backend.retrieval.repo_ranker import RepoRanker
from backend.retrieval.code_extractor import CodeExtractor
from backend.retrieval.huggingface_retrieval import HuggingFaceRetriever

__all__ = [
    "EmbeddingModel",
    "VectorStore",
    "RAGPipeline",
    "GitHubSearcher",
    "RepoRanker",
    "CodeExtractor",
    "HuggingFaceRetriever",
]

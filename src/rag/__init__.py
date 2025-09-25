"""
RAG (Retrieval-Augmented Generation) package for general document corpus.

This package provides modular components for:
- Document retrieval from ChromaDB
- Answer generation using LLMs
- Complete RAG pipeline orchestration
"""

from .retriever import DocumentRetriever
from .generator import AnswerGenerator
from .pipeline import RAGPipeline

__all__ = ['DocumentRetriever', 'AnswerGenerator', 'RAGPipeline']
"""
Abstract base classes for RAG pipeline components.

This module defines the core interfaces that all RAG pipeline components must implement
to ensure consistency, maintainability, and interoperability across the system.

Classes:
    BaseRetriever: Abstract interface for document retrieval components
    BaseGenerator: Abstract interface for answer generation components
    BaseEvaluator: Abstract interface for evaluation components
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from langchain.schema import Document
from omegaconf import DictConfig


class BaseRetriever(ABC):
    """
    Abstract base class for document retrieval components.
    
    All document retrievers must implement these core methods to ensure
    consistent behavior across different retrieval strategies and backends.
    """
    
    @abstractmethod
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Initialize the retriever with configuration.
        
        Args:
            config: System configuration object
            **kwargs: Additional initialization parameters
        """
        pass
    
    @abstractmethod
    def retrieve_documents(self, 
                          query: str, 
                          k: int = 5, 
                          **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The search query string
            k: Number of documents to retrieve
            **kwargs: Additional retrieval parameters (filters, thresholds, etc.)
            
        Returns:
            List of relevant Document objects with metadata
            
        Raises:
            RetrievalError: When retrieval operation fails
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying document collection.
        
        Returns:
            Dictionary containing collection statistics and metadata
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the retrieval system.
        
        Returns:
            Dictionary containing health status and diagnostic information
        """
        pass


class BaseGenerator(ABC):
    """
    Abstract base class for answer generation components.
    
    All answer generators must implement these core methods to ensure
    consistent behavior across different LLM providers and generation strategies.
    """
    
    @abstractmethod
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Initialize the generator with configuration.
        
        Args:
            config: System configuration object
            **kwargs: Additional initialization parameters
        """
        pass
    
    @abstractmethod
    def generate_answer(self, 
                       query: str, 
                       context_documents: List[Document],
                       **kwargs) -> Dict[str, Any]:
        """
        Generate an answer based on query and retrieved documents.
        
        Args:
            query: The user's question
            context_documents: List of relevant documents for context
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dictionary containing the generated answer and metadata
            
        Raises:
            GenerationError: When answer generation fails
        """
        pass
    
    @abstractmethod
    async def generate_streaming_answer(self, 
                                      query: str, 
                                      context_documents: List[Document],
                                      **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming answer based on query and retrieved documents.
        
        Args:
            query: The user's question
            context_documents: List of relevant documents for context
            **kwargs: Additional generation parameters
            
        Yields:
            Dictionary chunks containing partial answers and metadata
            
        Raises:
            GenerationError: When streaming generation fails
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the generation system.
        
        Returns:
            Dictionary containing health status and model information
        """
        pass


class BaseEvaluator(ABC):
    """
    Abstract base class for RAG evaluation components.
    
    All evaluators must implement these core methods to ensure
    consistent evaluation metrics and reporting across different evaluation strategies.
    """
    
    @abstractmethod
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: System configuration object
            **kwargs: Additional initialization parameters
        """
        pass
    
    @abstractmethod
    def evaluate_pipeline(self, 
                         test_questions: List[Dict[str, Any]],
                         **kwargs) -> Dict[str, Any]:
        """
        Evaluate the complete RAG pipeline on a set of test questions.
        
        Args:
            test_questions: List of questions with expected answers
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics and results
            
        Raises:
            EvaluationError: When evaluation process fails
        """
        pass
    
    @abstractmethod
    def evaluate_retrieval(self, 
                          test_queries: List[str],
                          **kwargs) -> Dict[str, Any]:
        """
        Evaluate retrieval component performance.
        
        Args:
            test_queries: List of test queries
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing retrieval metrics
        """
        pass
    
    @abstractmethod
    def evaluate_generation(self, 
                           query_context_pairs: List[Tuple[str, List[Document]]],
                           **kwargs) -> Dict[str, Any]:
        """
        Evaluate generation component performance.
        
        Args:
            query_context_pairs: List of (query, context_documents) pairs
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing generation quality metrics
        """
        pass


class RAGComponentError(Exception):
    """Base exception for RAG component errors."""
    pass


class RetrievalError(RAGComponentError):
    """Exception raised when document retrieval fails."""
    pass


class GenerationError(RAGComponentError):
    """Exception raised when answer generation fails."""
    pass


class EvaluationError(RAGComponentError):
    """Exception raised when evaluation process fails."""
    pass


class ConfigurationError(RAGComponentError):
    """Exception raised when configuration validation fails."""
    pass
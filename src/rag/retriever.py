"""
Document retrieval component for RAG pipeline.
Handles semantic search and document ranking from ChromaDB.
"""

import logging
from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from ..utils import load_config, preprocess_query
from ..interfaces.base_interfaces import BaseRetriever, RetrievalError
from ..utils.validation import validate_runtime_parameters, ParameterValidator
from ..utils.performance import time_operation, performance_context

logger = logging.getLogger(__name__)


class DocumentRetriever(BaseRetriever):
    """Handles document retrieval from ChromaDB vectorstore."""
    
    def __init__(self, 
                 config_path: str = "conf/config.yaml",
                 db_path: str = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 default_k: int = 5):
        """
        Initialize the document retriever.
        
        Args:
            config_path: Path to configuration file
            db_path: Override for ChromaDB storage path
            collection_name: Override for ChromaDB collection name
            embedding_model: Override for HuggingFace embedding model
            default_k: Default number of documents to retrieve
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Use config values or provided overrides
        embeddings_config = self.config.get('embeddings', {})
        vectordb_config = self.config.get('vectordb', {})
        
        self.db_path = db_path or vectordb_config.get('path', 'vector_store')
        self.collection_name = collection_name or vectordb_config.get('collection_name', 'rag_collection')
        embedding_model = embedding_model or embeddings_config.get('model', 'BAAI/bge-large-en-v1.5')
        
        # Check for container environment override
        import os
        if os.getenv("FORCE_CPU_DEVICE", "false").lower() == "true":
            embedding_device = 'cpu'
            logger.info("Container environment detected - forcing CPU for embeddings")
        else:
            embedding_device = embeddings_config.get('device', 'cpu')
        self.default_k = default_k
        
        # Reranking configuration
        self.reranking_enabled = embeddings_config.get('reranking_enabled', False)
        self.reranking_model = None
        
        if self.reranking_enabled:
            try:
                from sentence_transformers import CrossEncoder
                reranking_model_name = embeddings_config.get('reranking_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.reranking_model = CrossEncoder(reranking_model_name)
                logger.info(f"Initialized reranking model: {reranking_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available for reranking. Install with: pip install sentence-transformers")
                self.reranking_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize reranking model: {e}")
                self.reranking_enabled = False
        
        # Debug: Log the resolved values
        logger.info(f"Resolved config - DB path: {self.db_path}, Collection: {self.collection_name}")
        logger.info(f"Embedding model: {embedding_model}, Device: {embedding_device}")
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': embedding_device}
        )
        
        # Initialize vectorstore
        logger.info(f"Connecting to ChromaDB at {self.db_path}")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": default_k}
        )
        
        logger.info("Document retriever initialized successfully")
    
    @time_operation("collection_info_retrieval")
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector database collection.
        
        Returns:
            Dictionary containing collection statistics including:
            - total_documents: Total number of documents in collection
            - file_types: List of file types present
            - file_type_counts: Count of documents per file type
            - sample_sources: Sample of document sources
            
        Raises:
            RetrievalError: When collection info retrieval fails
        """
        try:
            logger.debug("Retrieving collection information")
            
            collection = self.vectorstore._collection
            count = collection.count()
            
            if count == 0:
                logger.warning("Vector database collection is empty")
                return {
                    'total_documents': 0,
                    'file_types': [],
                    'file_type_counts': {},
                    'sample_sources': []
                }
            
            logger.debug(f"Processing metadata for {count} documents")
            
            # Get all metadata to ensure accurate file type statistics
            with performance_context("metadata_analysis", document_count=count):
                results = collection.get(limit=count)
                file_type_counts = {}
                sources = set()
                
                for metadata in results.get('metadatas', []):
                    if metadata:
                        file_type = metadata.get('file_type', 'unknown')
                        file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
                        sources.add(metadata.get('citation_source', 'unknown'))
            
            collection_info = {
                'total_documents': count,
                'file_types': list(file_type_counts.keys()),
                'file_type_counts': file_type_counts,
                'sample_sources': list(sources)[:10]
            }
            
            logger.info(f"Collection info retrieved: {count} documents, {len(file_type_counts)} file types")
            return collection_info
            
        except Exception as e:
            error_context = {
                "collection_name": getattr(self, 'collection_name', 'unknown'),
                "db_path": getattr(self, 'db_path', 'unknown'),
                "vectorstore_type": type(getattr(self, 'vectorstore', None)).__name__
            }
            logger.error(f"Failed to get collection info: {str(e)}", extra={"context": error_context})
            raise RetrievalError(f"Failed to get collection information: {str(e)}") from e
    
    @time_operation("document_retrieval")
    @ParameterValidator.validate_query_parameters
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query with preprocessing.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve (defaults to default_k)
            
        Returns:
            List of relevant documents with metadata
            
        Raises:
            RetrievalError: When document retrieval fails
            ValueError: When parameters are invalid
            
        Example:
            >>> retriever = DocumentRetriever()
            >>> docs = retriever.retrieve_documents("machine learning", k=5)
            >>> print(f"Found {len(docs)} documents")
        """
        try:
            k = k or self.default_k
            
            # Validate parameters
            if not isinstance(query, str) or len(query.strip()) == 0:
                raise ValueError("Query must be a non-empty string")
            
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be a positive integer")
            
            logger.info(f"Retrieving documents for query: '{query[:100]}...', k={k}")
            
            with performance_context("query_preprocessing", query_length=len(query)):
                # Apply query preprocessing to improve retrieval quality
                processed_query = preprocess_query(query, config_path="conf/config.yaml")
                if processed_query != query:
                    logger.debug(f"Preprocessed query: '{query}' -> '{processed_query}'")
            
            with performance_context("vector_search", k=k, processed_query_length=len(processed_query)):
                docs = self.vectorstore.similarity_search(processed_query, k=k)
            
            logger.info(f"Successfully retrieved {len(docs)} documents")
            return docs
            
        except Exception as e:
            error_context = {
                "query": query[:200] if query else "None",
                "k": k,
                "vectorstore_initialized": hasattr(self, 'vectorstore'),
                "collection_name": getattr(self, 'collection_name', 'unknown')
            }
            logger.error(f"Document retrieval failed: {str(e)}", extra={"context": error_context})
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}") from e
    
    @time_operation("retriever_health_check")
    def health_check(self) -> Dict[str, Any]:
        """
        Check retriever health status and component connectivity.
        
        Returns:
            Dictionary containing health status and diagnostic information:
            - status: Overall health status (healthy/unhealthy/degraded)
            - vectorstore: Vector database connection status
            - collection_info: Collection statistics if available
            - diagnostics: Detailed diagnostic information
            
        Example:
            >>> retriever = DocumentRetriever()
            >>> health = retriever.health_check()
            >>> print(f"Status: {health['status']}")
        """
        import time
        
        diagnostics = {
            "timestamp": time.time(),
            "component": "DocumentRetriever",
            "version": "1.0.0"
        }
        
        try:
            logger.debug("Performing retriever health check")
            
            # Test vectorstore connection
            with performance_context("vectorstore_health_check"):
                collection_info = self.get_collection_info()
            
            # Determine health status
            if 'error' in collection_info:
                status = "unhealthy"
                diagnostics["vectorstore_error"] = collection_info['error']
            elif collection_info.get('total_documents', 0) == 0:
                status = "degraded"
                diagnostics["warning"] = "Vector database is empty"
            else:
                status = "healthy"
            
            health_result = {
                "status": status,
                "vectorstore": "connected" if 'error' not in collection_info else "disconnected",
                "collection_info": collection_info,
                "diagnostics": diagnostics
            }
            
            logger.info(f"Health check completed - Status: {status}")
            return health_result
            
        except Exception as e:
            error_context = {
                "component": "DocumentRetriever",
                "error_type": type(e).__name__,
                "db_path": getattr(self, 'db_path', 'unknown'),
                "collection_name": getattr(self, 'collection_name', 'unknown')
            }
            logger.error(f"Health check failed: {str(e)}", extra={"context": error_context})
            
            return {
                "status": "unhealthy",
                "vectorstore": "error",
                "error": str(e),
                "diagnostics": {**diagnostics, "error_context": error_context}
            }
    
    def retrieve_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores and preprocessing.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            k = k or self.default_k
            
            # Apply query preprocessing to improve retrieval quality
            processed_query = preprocess_query(query, config_path="conf/config.yaml")
            if processed_query != query:
                logger.debug(f"Preprocessed query for scoring: '{query}' -> '{processed_query}'")
            
            docs_with_scores = self.vectorstore.similarity_search_with_score(processed_query, k=k)
            logger.debug(f"Retrieved {len(docs_with_scores)} documents with scores")
            
            # Apply reranking if enabled
            if self.reranking_enabled:
                reranking_top_k = self.config.get('embeddings', {}).get('reranking_top_k', k)
                docs_with_scores = self._rerank_documents(processed_query, docs_with_scores, reranking_top_k)
            
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {e}")
            return []
    
    def retrieve_by_file_type(self, query: str, file_type: str, k: int = None) -> List[Document]:
        """
        Retrieve documents filtered by file type.
        
        Args:
            query: Search query
            file_type: File type to filter by (html, pdf, docx, etc.)
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents of specified type
        """
        try:
            k = k or self.default_k
            
            # Use metadata filtering
            docs = self.vectorstore.similarity_search(
                query, 
                k=k*2,  # Get more results to account for filtering
                filter={"file_type": file_type}
            )
            
            # Take only the requested number
            docs = docs[:k]
            logger.debug(f"Retrieved {len(docs)} {file_type} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving {file_type} documents: {e}")
            return []
    
    def get_source_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get summary information about retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Summary statistics about the documents
        """
        if not documents:
            return {'num_documents': 0, 'file_types': [], 'sources': []}
        
        file_types = []
        sources = []
        
        for doc in documents:
            file_types.append(doc.metadata.get('file_type', 'unknown'))
            sources.append(doc.metadata.get('citation_source', 'unknown'))
        
        return {
            'num_documents': len(documents),
            'file_types': list(set(file_types)),
            'sources': list(set(sources)),
            'file_type_counts': {ft: file_types.count(ft) for ft in set(file_types)}
        }
    
    def calculate_retrieval_confidence(self, docs_with_scores: List[tuple], query: str) -> Dict[str, Any]:
        """
        Calculate confidence metrics for retrieved documents.
        
        Args:
            docs_with_scores: List of (Document, similarity_score) tuples
            query: Original search query
            
        Returns:
            Dictionary with confidence metrics and retrieval metadata
        """
        if not docs_with_scores:
            return {
                "confidence_score": 0.0,
                "confidence_level": "very_low",
                "retrieval_quality": "poor",
                "num_documents_found": 0,
                "best_match_score": 0.0,
                "score_distribution": [],
                "diverse_sources": False
            }
        
        # Extract scores (ChromaDB uses distance, lower is better)
        scores = [score for _, score in docs_with_scores]
        best_score = min(scores)  # Lower distance = better match
        avg_score = sum(scores) / len(scores)
        
        # Convert distance to similarity (inverse relationship)
        # Typical ChromaDB distances range from 0.0 (perfect) to 2.0 (very different)
        best_similarity = max(0, 1.0 - (best_score / 2.0))
        avg_similarity = max(0, 1.0 - (avg_score / 2.0))
        
        # Calculate confidence based on best match and consistency
        confidence_score = (best_similarity * 0.7) + (avg_similarity * 0.3)
        
        # Determine confidence levels
        if confidence_score >= 0.8:
            confidence_level = "high"
            retrieval_quality = "excellent"
        elif confidence_score >= 0.6:
            confidence_level = "medium"
            retrieval_quality = "good"
        elif confidence_score >= 0.4:
            confidence_level = "low"
            retrieval_quality = "fair"
        else:
            confidence_level = "very_low"
            retrieval_quality = "poor"
        
        # Check source diversity
        sources = set()
        file_types = set()
        for doc, _ in docs_with_scores:
            sources.add(doc.metadata.get('citation_source', 'unknown'))
            file_types.add(doc.metadata.get('file_type', 'unknown'))
        
        diverse_sources = len(sources) > len(docs_with_scores) // 2
        
        return {
            "confidence_score": round(confidence_score, 3),
            "confidence_level": confidence_level,
            "retrieval_quality": retrieval_quality,
            "num_documents_found": len(docs_with_scores),
            "best_match_score": round(best_similarity, 3),
            "avg_match_score": round(avg_similarity, 3),
            "score_distribution": [round(1.0 - (s/2.0), 3) for s in scores],
            "diverse_sources": diverse_sources,
            "unique_sources": len(sources),
            "file_types_found": list(file_types),
            "query_length": len(query.split())
        }
    
    def _rerank_documents(self, query: str, docs_with_scores: List[Tuple[Document, float]], top_k: int = None) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder for better relevance."""
        if not self.reranking_enabled or not self.reranking_model:
            return docs_with_scores
        
        if not docs_with_scores:
            return docs_with_scores
        
        try:
            # Prepare query-document pairs for reranking
            pairs = []
            docs = []
            for doc, _ in docs_with_scores:
                pairs.append([query, doc.page_content[:512]])  # Limit content for reranking
                docs.append(doc)
            
            # Get reranking scores
            rerank_scores = self.reranking_model.predict(pairs)
            
            # Combine documents with new scores
            reranked = list(zip(docs, rerank_scores))
            
            # Sort by reranking score (higher is better)
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            # Apply top_k if specified
            if top_k:
                reranked = reranked[:top_k]
            
            logger.debug(f"Reranked {len(docs_with_scores)} documents, returning top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return docs_with_scores
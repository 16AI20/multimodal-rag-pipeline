"""
Main RAG pipeline orchestrator.
Combines retrieval and generation components for complete RAG functionality.
"""

import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from langchain.schema import Document

from .retriever import DocumentRetriever
from .generator import AnswerGenerator
from ..observability import LangfuseTracker
from ..utils.async_helpers import async_retry, concurrent_map, make_async
from ..utils.performance import time_operation, performance_context

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""
    
    def __init__(self,
                 config_path: str = "conf",
                 db_path: str = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 llm_model: str = None,
                 default_k: int = 5,
                 temperature: float = None):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            config_path: Path to configuration directory
            db_path: Override for ChromaDB storage path
            collection_name: Override for ChromaDB collection name
            embedding_model: Override for HuggingFace embedding model
            llm_model: Override for language model
            default_k: Default number of documents to retrieve
            temperature: Override for LLM sampling temperature
        """
        logger.info("Initializing RAG pipeline...")
        
        # Load configuration for observability
        from ..utils import load_config
        self.config = load_config(config_path)
        
        # Initialize Langfuse tracker
        self.langfuse_tracker = LangfuseTracker(self.config)
        
        # Initialize retriever
        self.retriever = DocumentRetriever(
            config_path=config_path,
            db_path=db_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            default_k=default_k
        )
        
        # Initialize generator
        self.generator = AnswerGenerator(
            config_path=config_path,
            llm_model=llm_model,
            temperature=temperature
        )
        
        # Create RAG chain for legacy compatibility
        self.rag_chain = self.generator.create_rag_chain(self.retriever.retriever)
        
        logger.info("RAG pipeline initialized successfully!")
    
    @async_retry(max_attempts=3, delay=1.0)
    async def async_query(self, 
                         question: str, 
                         k: int = None,
                         file_type: str = None,
                         return_sources: bool = True,
                         user_id: str = None,
                         session_id: str = None) -> Dict[str, Any]:
        """
        Process a complete RAG query asynchronously with retry logic.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            file_type: Optional file type filter
            return_sources: Whether to include source information
            user_id: Optional user identifier for tracking
            session_id: Optional session identifier for tracking
            
        Returns:
            Complete RAG response with answer and sources
            
        Raises:
            RetrievalError: When document retrieval fails
            GenerationError: When answer generation fails
        """
        logger.info(f"Processing async query: {question[:50]}...")
        
        with performance_context("async_rag_query", question_length=len(question), k=k):
            # Create async versions of sync operations
            async_retrieve = make_async(self.retriever.retrieve_documents)
            async_generate = make_async(self.generator.generate_answer)
            
            # Execute retrieval and any additional async operations
            with performance_context("async_retrieval"):
                documents = await async_retrieve(question, k=k)
            
            # Execute generation
            with performance_context("async_generation"):
                result = await async_generate(question, documents)
            
            # Add timing information
            result["async_processing"] = True
            
            logger.info(f"Async query completed successfully")
            return result
    
    @time_operation("rag_pipeline_query")
    def query(self, 
              question: str, 
              k: int = None,
              file_type: str = None,
              return_sources: bool = True,
              user_id: str = None,
              session_id: str = None) -> Dict[str, Any]:
        """
        Process a complete RAG query.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            file_type: Optional file type filter
            return_sources: Whether to include source information
            user_id: Optional user identifier for tracking
            session_id: Optional session identifier for tracking
            
        Returns:
            Complete RAG response with answer and sources
        """
        query_start_time = time.time()
        
        # Start Langfuse trace
        with self.langfuse_tracker.trace_query(question, user_id, session_id) as trace:
            try:
                logger.info(f"Processing query: {question[:50]}...")
                
                # Retrieval phase
                retrieval_start_time = time.time()
                
                if file_type:
                    documents = self.retriever.retrieve_by_file_type(question, file_type, k)
                    docs_with_scores = [(doc, 0.5) for doc in documents]  # Fallback scores for filtered results
                else:
                    docs_with_scores = self.retriever.retrieve_with_scores(question, k)
                    documents = [doc for doc, _ in docs_with_scores]
                
                retrieval_time = time.time() - retrieval_start_time
                
                # Track retrieval with Langfuse
                docs_for_tracking = [
                    {
                        'source': doc.metadata.get('citation_source', ''),
                        'file_type': doc.metadata.get('file_type', ''),
                        'content_preview': doc.page_content[:200],
                        'similarity_score': score
                    } for doc, score in docs_with_scores[:10]  # Limit for tracking
                ]
                self.langfuse_tracker.track_retrieval(trace, question, docs_for_tracking, retrieval_time, k)
                
                # Calculate retrieval confidence
                retrieval_metadata = self.retriever.calculate_retrieval_confidence(docs_with_scores, question)
            
                if not documents:
                    result = {
                        'query': question,
                        'answer': "I couldn't find any relevant information to answer your question.",
                        'sources': [],
                        'num_sources': 0,
                        'retrieval_info': {
                            'documents_found': 0,
                            'file_types': [],
                            'search_filter': file_type
                        },
                        'retrieval_metadata': retrieval_metadata
                    }
                    
                    # Track empty result
                    if trace:
                        trace.update(
                            output=result,
                            metadata={"total_time": time.time() - query_start_time}
                        )
                    
                    return result
                
                # Generation phase
                generation_start_time = time.time()
                result = self.generator.generate_from_documents(question, documents)
                generation_time = time.time() - generation_start_time
                
                # Track generation with Langfuse
                if result.get('answer'):
                    self.langfuse_tracker.track_generation(
                        trace, 
                        question, 
                        docs_for_tracking, 
                        result['answer'], 
                        generation_time,
                        model=getattr(self.generator, 'llm_model', None)
                    )
            
                # Add retrieval information
                if return_sources:
                    retrieval_info = self.retriever.get_source_summary(documents)
                    retrieval_info['search_filter'] = file_type
                    result['retrieval_info'] = retrieval_info
                
                # Always add confidence metadata for transparency
                result['retrieval_metadata'] = retrieval_metadata
                
                # Update trace with final result
                if trace:
                    total_time = time.time() - query_start_time
                    trace.update(
                        output=result,
                        metadata={
                            "total_time": total_time,
                            "retrieval_time": retrieval_time,
                            "generation_time": generation_time,
                            "num_documents": len(documents),
                            "response_length": len(result.get('answer', ''))
                        }
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                
                # Track error with Langfuse
                self.langfuse_tracker.track_error(trace, e, "query_processing")
                
                error_result = {
                    'query': question,
                    'error': str(e),
                    'answer': None,
                    'sources': []
                }
                
                if trace:
                    trace.update(
                        level="ERROR",
                        status_message=str(e),
                        output=error_result,
                        metadata={"total_time": time.time() - query_start_time}
                    )
                
                return error_result
    
    def retrieve_only(self, 
                     question: str, 
                     k: int = None,
                     file_type: str = None,
                     include_scores: bool = False) -> Dict[str, Any]:
        """
        Perform retrieval only (no generation).
        
        Args:
            question: Search query
            k: Number of documents to retrieve
            file_type: Optional file type filter
            include_scores: Whether to include similarity scores
            
        Returns:
            Retrieval results with document information
        """
        try:
            # Choose retrieval method
            if include_scores:
                docs_with_scores = self.retriever.retrieve_with_scores(question, k)
                documents = [doc for doc, _ in docs_with_scores]
                scores = [score for _, score in docs_with_scores]
            else:
                if file_type:
                    documents = self.retriever.retrieve_by_file_type(question, file_type, k)
                else:
                    documents = self.retriever.retrieve_documents(question, k)
                scores = None
            
            # Prepare response
            response = {
                'query': question,
                'documents': [],
                'num_documents': len(documents),
                'retrieval_info': self.retriever.get_source_summary(documents)
            }
            
            # Add document details
            for i, doc in enumerate(documents):
                doc_info = {
                    'rank': i + 1,
                    'source': doc.metadata.get('citation_source', 'Unknown'),
                    'file_type': doc.metadata.get('file_type', 'unknown'),
                    'content_preview': (
                        doc.page_content[:200] + "..." 
                        if len(doc.page_content) > 200 
                        else doc.page_content
                    )
                }
                
                if scores:
                    doc_info['similarity_score'] = float(scores[i])
                
                response['documents'].append(doc_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return {
                'query': question,
                'error': str(e),
                'documents': [],
                'num_documents': 0
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector database collection."""
        return self.retriever.get_collection_info()
    
    def health_check(self, skip_generator_test: bool = False) -> Dict[str, Any]:
        """
        Perform a health check of the pipeline components.
        
        Args:
            skip_generator_test: Skip the LLM test call to avoid extra HTTP requests
        
        Returns:
            Health status of retriever and generator
        """
        try:
            # Check retriever
            collection_info = self.retriever.get_collection_info()
            retriever_healthy = 'error' not in collection_info and collection_info.get('total_documents', 0) > 0
            
            # Check generator - can skip test call to avoid duplicate HTTP requests
            if skip_generator_test:
                # Basic generator check without making an LLM call
                generator_healthy = hasattr(self.generator, 'llm') and self.generator.llm is not None
                generator_status = 'healthy (not tested)' if generator_healthy else 'unhealthy'
            else:
                # Full generator test with LLM call
                test_docs = [Document(page_content="Test content", metadata={'citation_source': 'test'})]
                test_result = self.generator.generate_from_documents("test", test_docs)
                generator_healthy = 'error' not in test_result
                generator_status = 'healthy' if generator_healthy else 'unhealthy'
            
            return {
                'status': 'healthy' if retriever_healthy and generator_healthy else 'unhealthy',
                'components': {
                    'retriever': 'healthy' if retriever_healthy else 'unhealthy',
                    'generator': generator_status
                },
                'collection_info': collection_info
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def generate_streaming_answer(self, 
                                      question: str, 
                                      documents: List[Dict[str, Any]], 
                                      temperature: float = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer from retrieved documents.
        Uses the same quality logic as the regular query method.
        
        Args:
            question: User's question
            documents: Retrieved documents with metadata
            temperature: Override for LLM temperature
            
        Yields:
            Tokens as they are generated
        """
        try:
            # Convert document dicts to Document objects (same as regular query)
            doc_objects = []
            for doc in documents:
                doc_objects.append(Document(
                    page_content=doc.get('content', ''),
                    metadata={
                        'citation_source': doc.get('source', ''),
                        'file_type': doc.get('file_type', ''),
                        'rank': doc.get('rank', 0)
                    }
                ))
            
            # Use generator's streaming method
            async for token in self.generator.generate_streaming_from_documents(
                question, doc_objects, temperature
            ):
                yield token
                
        except Exception as e:
            logger.error("Error in streaming answer generation: %s", e)
            logger.debug("Streaming generation error details", exc_info=True)
            yield f"Error generating streaming response: {str(e)}"

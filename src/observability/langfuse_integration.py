"""
Langfuse integration for RAG pipeline observability.
Provides tracking for queries, retrievals, generations, and performance metrics.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Optional Langfuse import - gracefully handle if not installed
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning("Langfuse not installed. Install with: pip install langfuse")
    LANGFUSE_AVAILABLE = False
    
    # Create dummy decorators for when Langfuse is not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class QueryMetrics:
    """Metrics for a single RAG query."""
    query_id: str
    question: str
    response_time: float
    retrieval_time: float
    generation_time: float
    num_documents_retrieved: int
    num_sources: int
    response_length: int
    token_count: Optional[int] = None
    cost: Optional[float] = None
    category: Optional[str] = None
    user_feedback: Optional[int] = None  # 1-5 rating
    error: Optional[str] = None


class LangfuseTracker:
    """Langfuse integration for RAG pipeline observability."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize Langfuse tracker with configuration.
        
        Args:
            config: Hydra configuration containing observability settings.
        """
        # Safely get observability config or use empty dict
        self.config = getattr(config, 'observability', {})
        self.enabled = self.config.get('enabled', False) if self.config else False
        self.langfuse = None
        
        if self.enabled and LANGFUSE_AVAILABLE:
            self._initialize_langfuse()
        elif self.enabled and not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse tracking enabled in config but langfuse package not installed")
            self.enabled = False
        
        logger.info(f"Langfuse tracking: {'enabled' if self.enabled else 'disabled'}")
    
    def _initialize_langfuse(self) -> None:
        """Initialize Langfuse client with configuration.
        
        Raises:
            Exception: If Langfuse initialization fails.
        """
        try:
            # Get credentials from environment or config
            public_key = os.getenv('LANGFUSE_PUBLIC_KEY') or (self.config.get('public_key', '') if self.config else '')
            secret_key = os.getenv('LANGFUSE_SECRET_KEY') or (self.config.get('secret_key', '') if self.config else '')
            host = self.config.get('host', 'http://localhost:3000') if self.config else 'http://localhost:3000'
            
            if not public_key or not secret_key:
                logger.warning("Langfuse credentials not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
                self.enabled = False
                return
            
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=(self.config.get('advanced', {}).get('debug', False) if self.config else False)
            )
            
            # Test connection
            self.langfuse.auth_check()
            logger.info("Langfuse connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False
    
    @contextmanager
    def trace_query(self, question: str, user_id: str = None, session_id: str = None):
        """Context manager for tracing a complete RAG query."""
        if not self.enabled:
            yield None
            return
        
        trace = None
        try:
            # Create trace for the complete query
            trace = self.langfuse.trace(
                name="rag_query",
                input={"question": question},
                user_id=user_id,
                session_id=session_id,
                tags=list(self.config.get('tags', {}).values()) if self.config and self.config.get('tags') else [],
                metadata={
                    "project": self.config.get('tags', {}).get('project', 'rag-system') if self.config else 'rag-system',
                    "environment": self.config.get('tags', {}).get('environment', 'development') if self.config else 'development',
                    "version": self.config.get('tags', {}).get('version', '1.0.0') if self.config else '1.0.0'
                }
            )
            
            yield trace
            
        except Exception as e:
            logger.error(f"Error in Langfuse trace: {e}")
            if trace:
                trace.update(level="ERROR", status_message=str(e))
            yield None
        finally:
            if trace:
                self.langfuse.flush()
    
    def track_retrieval(self, trace, question: str, documents: List[Dict], 
                       retrieval_time: float, k: int = None) -> None:
        """Track document retrieval step.
        
        Args:
            trace: Langfuse trace object.
            question: User's question.
            documents: Retrieved documents with metadata.
            retrieval_time: Time taken for retrieval in seconds.
            k: Number of documents requested.
        """
        if not self.enabled or not trace:
            return
        
        try:
            # Calculate retrieval metrics
            num_retrieved = len(documents)
            source_types = [doc.get('file_type', 'unknown') for doc in documents]
            source_distribution = {ft: source_types.count(ft) for ft in set(source_types)}
            
            # Average similarity score if available
            avg_similarity = None
            similarities = [doc.get('similarity_score') for doc in documents if doc.get('similarity_score')]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
            
            trace.span(
                name="retrieval",
                input={
                    "question": question,
                    "k": k or num_retrieved
                },
                output={
                    "num_documents": num_retrieved,
                    "source_distribution": source_distribution,
                    "avg_similarity": avg_similarity
                },
                metadata={
                    "retrieval_time": retrieval_time,
                    "documents": [
                        {
                            "source": doc.get('source', ''),
                            "file_type": doc.get('file_type', ''),
                            "similarity": doc.get('similarity_score'),
                            "preview": doc.get('content_preview', '')[:200]
                        } for doc in documents[:5]  # Only first 5 for brevity
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Error tracking retrieval: {e}")
    
    def track_generation(self, trace, question: str, documents: List[Dict], 
                        response: str, generation_time: float, 
                        token_count: int = None, model: str = None) -> None:
        """Track answer generation step.
        
        Args:
            trace: Langfuse trace object.
            question: User's question.
            documents: Context documents used for generation.
            response: Generated response text.
            generation_time: Time taken for generation in seconds.
            token_count: Optional token count for cost estimation.
            model: Optional model name used for generation.
        """
        if not self.enabled or not trace:
            return
        
        try:
            # Calculate generation metrics
            response_length = len(response)
            word_count = len(response.split())
            
            # Estimate cost if token count available (rough estimates)
            estimated_cost = None
            if token_count and model:
                # Very rough cost estimates - adjust based on actual pricing
                cost_per_1k_tokens = {
                    'gpt-3.5-turbo': 0.002,
                    'gpt-4': 0.03,
                    'llama3.1:8b': 0.0,  # Local/free models
                }.get(model, 0.001)
                estimated_cost = (token_count / 1000) * cost_per_1k_tokens
            
            generation_span = trace.span(
                name="generation",
                input={
                    "question": question,
                    "context_docs": len(documents),
                    "model": model
                },
                output={
                    "response": response,
                    "response_length": response_length,
                    "word_count": word_count
                },
                metadata={
                    "generation_time": generation_time,
                    "token_count": token_count,
                    "estimated_cost": estimated_cost,
                    "model": model
                }
            )
            
            # Track token usage if available
            if token_count:
                generation_span.update(usage={
                    "total_tokens": token_count,
                    "estimated_cost": estimated_cost
                })
            
        except Exception as e:
            logger.error(f"Error tracking generation: {e}")
    
    def track_error(self, trace, error: Exception, step: str = "unknown") -> None:
        """Track errors in the RAG pipeline.
        
        Args:
            trace: Langfuse trace object.
            error: Exception that occurred.
            step: Step where error occurred.
        """
        if not self.enabled or not trace:
            return
        
        try:
            trace.span(
                name=f"error_{step}",
                level="ERROR",
                status_message=str(error),
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "step": step
                }
            )
            
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
    
    def track_user_feedback(self, trace_id: str, rating: int, comment: str = None) -> None:
        """Track user feedback for a query.
        
        Args:
            trace_id: ID of the trace to add feedback to.
            rating: User rating (1-5 scale).
            comment: Optional user comment.
        """
        if not self.enabled:
            return
        
        try:
            # Update trace with user feedback
            self.langfuse.score(
                trace_id=trace_id,
                name="user_rating",
                value=rating,
                comment=comment
            )
            
        except Exception as e:
            logger.error(f"Error tracking user feedback: {e}")
    
    def get_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get analytics data from Langfuse (if available).
        
        Args:
            time_range: Time range for analytics (e.g., '24h', '7d').
            
        Returns:
            Dictionary containing analytics data or dashboard URL.
        """
        if not self.enabled:
            return {"error": "Langfuse tracking not enabled"}
        
        # Note: This would require Langfuse analytics API
        # For now, return placeholder
        return {
            "message": "Analytics available in Langfuse dashboard",
            "dashboard_url": f"{self.config.get('host', 'http://localhost:3000') if self.config else 'http://localhost:3000'}/project/{self.config.get('tags', {}).get('project', 'default') if self.config else 'default'}"
        }
    
    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")


# Decorator for tracking functions
def track_rag_operation(operation_name: str):
    """Decorator to track RAG operations with Langfuse."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the instance has a langfuse_tracker
            tracker = getattr(self, 'langfuse_tracker', None)
            if not tracker or not tracker.enabled:
                return func(self, *args, **kwargs)
            
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                end_time = time.time()
                
                # Log success metrics
                if hasattr(tracker, 'langfuse') and tracker.langfuse:
                    tracker.langfuse.span(
                        name=operation_name,
                        input={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                        output={"success": True},
                        metadata={"duration": end_time - start_time}
                    )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                
                # Log error
                if hasattr(tracker, 'langfuse') and tracker.langfuse:
                    tracker.langfuse.span(
                        name=operation_name,
                        level="ERROR",
                        status_message=str(e),
                        metadata={"duration": end_time - start_time}
                    )
                
                raise
        
        return wrapper
    return decorator
#!/usr/bin/env python
"""
FastAPI backend for RAG pipeline.
Provides REST API endpoints for retrieval and generation.

Usage:
    uvicorn src.interfaces.fastapi_app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health        - Health check
    POST /query         - Full RAG query
    POST /retrieve      - Retrieval only
    GET  /collection    - Collection info
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
import os
import signal
import sys
from typing import AsyncGenerator

# Set environment variables to prevent multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from ..rag import RAGPipeline
from ..api.schemas import (
    QueryRequest, QueryResponse, 
    RetrievalRequest, RetrievalResponse,
    HealthResponse, CollectionInfo, StreamChunk
)
from ..monitoring.health_monitor import setup_monitoring, create_health_endpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize RAG pipeline on startup."""
    global rag_pipeline
    
    logger.info("Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline()
        
        # Setup health monitoring
        health_monitor = setup_monitoring(
            rag_pipeline.retriever, 
            rag_pipeline.generator, 
            rag_pipeline
        )
        
        # Store health monitor globally for endpoint access
        app.state.health_monitor = health_monitor
        
        logger.info("RAG pipeline and monitoring initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down RAG pipeline...")
    if rag_pipeline:
        # Clean up any resources
        try:
            import gc
            
            # Try to clear CUDA cache if torch and CUDA are available
            try:
                import torch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Force garbage collection
            gc.collect()
            
        except Exception as cleanup_error:
            logger.warning(f"Cleanup warning: {cleanup_error}")
        
        rag_pipeline = None
        
        # Shutdown monitoring
        if hasattr(app.state, 'health_monitor'):
            app.state.health_monitor.stop_monitoring()


# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="REST API for Retrieval-Augmented Generation Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # React, Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health of the RAG pipeline with comprehensive monitoring.
    
    Returns:
        Health status of all RAG pipeline components including:
        - Overall system status
        - Individual component health (retriever, generator, vector db)
        - Performance metrics and response times
        - Active alerts and system diagnostics
        
    Raises:
        HTTPException: If health check fails or pipeline not initialized
        
    Example:
        GET /health
        {
            "overall_status": "healthy",
            "components": {...},
            "active_alerts": [],
            "system_metrics": {...}
        }
    """
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Use enhanced health monitoring if available
        if hasattr(app.state, 'health_monitor'):
            health_data = app.state.health_monitor.get_system_health()
            
            # Determine HTTP status based on overall health
            if health_data["overall_status"] == "unhealthy":
                raise HTTPException(status_code=503, detail=health_data)
            elif health_data["overall_status"] == "degraded":
                # Return 200 but include degraded status in response
                return HealthResponse(
                    status="degraded",
                    message="System is degraded but operational",
                    details=health_data
                )
            else:
                return HealthResponse(
                    status="healthy", 
                    message="All systems operational",
                    details=health_data
                )
        else:
            # Fallback to basic health check
            basic_health = {
                "status": "healthy",
                "pipeline_initialized": True,
                "timestamp": time.time()
            }
            return HealthResponse(**basic_health)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")


@app.get("/collection", response_model=CollectionInfo)
async def get_collection_info() -> CollectionInfo:
    """Get information about the vector database collection.
    
    Returns:
        Information about the vector database collection.
        
    Raises:
        HTTPException: If collection info retrieval fails.
    """
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        info = rag_pipeline.get_collection_info()
        if 'error' in info:
            raise HTTPException(status_code=500, detail=info['error'])
        
        return CollectionInfo(**info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_sse_stream(request: QueryRequest) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events stream for RAG query."""
    try:
        if not rag_pipeline:
            error_chunk = StreamChunk(type="error", error="RAG pipeline not initialized")
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            return
        
        # Do retrieval only (no generation) to get documents
        retrieval_result = rag_pipeline.retrieve_only(
            question=request.question,
            k=request.k,
            file_type=request.file_type,
            include_scores=True
        )
        
        if 'error' in retrieval_result:
            error_chunk = StreamChunk(type="error", error=retrieval_result['error'])
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            return
        
        # Prepare sources in the expected format
        sources = []
        documents = []
        for doc_info in retrieval_result.get('documents', []):
            source = {
                'citation': doc_info.get('source', ''),
                'file_type': doc_info.get('file_type', ''),
                'content_preview': doc_info.get('content_preview', '')
            }
            sources.append(source)
            
            # Also prepare for streaming generator
            documents.append({
                'content': doc_info.get('content_preview', ''),
                'source': doc_info.get('source', ''),
                'file_type': doc_info.get('file_type', ''),
                'rank': doc_info.get('rank', len(documents) + 1)
            })
        
        # Send sources chunk immediately
        sources_chunk = StreamChunk(
            type="sources",
            sources=sources,
            retrieval_info=retrieval_result.get('retrieval_info'),
            retrieval_metadata={}  # retrieve_only doesn't include this
        )
        yield f"data: {sources_chunk.model_dump_json()}\n\n"
        
        # Stream the answer using retrieved documents (single LLM call)
        full_answer = ""
        async for token in rag_pipeline.generate_streaming_answer(
            question=request.question,
            documents=documents,
            temperature=request.temperature
        ):
            if token:
                full_answer += token
                token_chunk = StreamChunk(type="token", content=token)
                yield f"data: {token_chunk.model_dump_json()}\n\n"
        
        # Send completion chunk
        complete_response = QueryResponse(
            query=request.question,
            answer=full_answer,
            sources=sources_chunk.sources or [],
            num_sources=len(sources_chunk.sources or []),
            retrieval_info=sources_chunk.retrieval_info,
            retrieval_metadata=sources_chunk.retrieval_metadata
        )
        
        complete_chunk = StreamChunk(type="complete", complete_response=complete_response)
        yield f"data: {complete_chunk.model_dump_json()}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        error_chunk = StreamChunk(type="error", error=str(e))
        yield f"data: {error_chunk.model_dump_json()}\n\n"


@app.post("/query")
async def rag_query(request: QueryRequest):
    """Process a complete RAG query with retrieval and generation.
    
    Args:
        request: Query request containing question and parameters.
        
    Returns:
        QueryResponse for non-streaming requests, StreamingResponse for streaming.
        
    Raises:
        HTTPException: If query processing fails.
    """
    # Handle streaming requests
    if request.stream:
        return StreamingResponse(
            generate_sse_stream(request),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    # Handle regular requests
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        result = rag_pipeline.query(
            question=request.question,
            k=request.k,
            file_type=request.file_type,
            return_sources=request.return_sources
        )
        
        if 'error' in result:
            logger.error(f"RAG pipeline error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Check if answer is None
        if result.get('answer') is None:
            logger.error(f"RAG pipeline returned None answer. Full result: {result}")
            raise HTTPException(status_code=500, detail="RAG pipeline returned None answer - likely LLM connection issue")
        
        return QueryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest) -> RetrievalResponse:
    """Retrieve relevant documents without generation.
    
    Args:
        request: Retrieval request containing query and parameters.
        
    Returns:
        Retrieved documents with metadata.
        
    Raises:
        HTTPException: If retrieval fails.
    """
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        result = rag_pipeline.retrieve_only(
            question=request.question,
            k=request.k,
            file_type=request.file_type,
            include_scores=request.include_scores
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return RetrievalResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information.
    
    Returns:
        Dictionary containing API information and available endpoints.
    """
    return {
        "name": "RAG Pipeline API",
        "version": "1.0.0",
        "description": "REST API for Retrieval-Augmented Generation Pipeline",
        "endpoints": {
            "health": "GET /health - Health check",
            "collection": "GET /collection - Collection information",
            "query": "POST /query - Full RAG query",
            "retrieve": "POST /retrieve - Retrieval only",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully.
    
    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    global rag_pipeline
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    if rag_pipeline:
        try:
            import gc
            # Force cleanup
            gc.collect()
            rag_pipeline = None
        except Exception as e:
            logger.warning(f"Cleanup warning during signal handling: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
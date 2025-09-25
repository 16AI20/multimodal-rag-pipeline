"""
Pydantic schemas for API request/response validation.
Defines the data models for RAG pipeline interactions.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    
    question: str = Field(..., description="The user's question", min_length=1)
    k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    file_type: Optional[str] = Field(None, description="Filter by file type (html, pdf, docx, etc.)")
    temperature: Optional[float] = Field(0.7, description="LLM temperature", ge=0.0, le=2.0)
    return_sources: bool = Field(True, description="Whether to include source information")
    stream: bool = Field(False, description="Whether to stream the response")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What topics are covered in the documents?",
                "k": 5,
                "file_type": None,
                "temperature": 0.7,
                "return_sources": True,
                "stream": False
            }
        }


class RetrievalRequest(BaseModel):
    """Request model for retrieval-only queries."""
    
    question: str = Field(..., description="The search query", min_length=1)
    k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    file_type: Optional[str] = Field(None, description="Filter by file type")
    include_scores: bool = Field(False, description="Whether to include similarity scores")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "machine learning techniques",
                "k": 5,
                "file_type": "pdf",
                "include_scores": True
            }
        }


class DocumentInfo(BaseModel):
    """Model for document information in responses."""
    
    rank: int = Field(..., description="Document rank in results")
    source: str = Field(..., description="Citation source")
    file_type: str = Field(..., description="File type")
    content_preview: str = Field(..., description="Preview of document content")
    similarity_score: Optional[float] = Field(None, description="Similarity score if available")


class SourceInfo(BaseModel):
    """Model for source information."""
    
    citation: str = Field(..., description="Citation source")
    file_type: str = Field(..., description="File type")
    content_preview: str = Field(..., description="Content preview")


class RetrievalInfo(BaseModel):
    """Model for retrieval metadata."""
    
    num_documents: int = Field(..., description="Number of documents found")
    file_types: List[str] = Field(..., description="File types in results")
    sources: List[str] = Field(..., description="Unique sources")
    file_type_counts: Dict[str, int] = Field(..., description="Count by file type")
    search_filter: Optional[str] = Field(None, description="Applied file type filter")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    
    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(..., description="Source documents used")
    num_sources: int = Field(..., description="Number of sources")
    retrieval_info: Optional[RetrievalInfo] = Field(None, description="Retrieval metadata")
    retrieval_metadata: Optional[Dict[str, Any]] = Field(None, description="Confidence and quality metrics")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the AI Apprenticeship Programme?",
                "answer": "The document corpus covers various topics including...",
                "sources": [
                    {
                        "citation": "Overview Document",
                        "file_type": "html",
                        "content_preview": "The document corpus contains information about..."
                    }
                ],
                "num_sources": 3,
                "retrieval_info": {
                    "num_documents": 3,
                    "file_types": ["html", "pdf"],
                    "sources": ["Overview Document", "Technical Details"],
                    "file_type_counts": {"html": 2, "pdf": 1},
                    "search_filter": None
                }
            }
        }


class RetrievalResponse(BaseModel):
    """Response model for retrieval-only queries."""
    
    query: str = Field(..., description="Original search query")
    documents: List[DocumentInfo] = Field(..., description="Retrieved documents")
    num_documents: int = Field(..., description="Number of documents retrieved")
    retrieval_info: RetrievalInfo = Field(..., description="Retrieval metadata")
    error: Optional[str] = Field(None, description="Error message if any")


class CollectionInfo(BaseModel):
    """Model for vector database collection information."""
    
    total_documents: int = Field(..., description="Total number of documents")
    file_types: List[str] = Field(..., description="Available file types")
    file_type_counts: Dict[str, int] = Field(..., description="Count of documents per file type")
    sample_sources: List[str] = Field(..., description="Sample source names")


class ComponentHealth(BaseModel):
    """Model for component health status."""
    
    retriever: str = Field(..., description="Retriever health status")
    generator: str = Field(..., description="Generator health status")


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    
    type: str = Field(..., description="Chunk type: 'sources', 'token', 'complete', 'error'")
    content: Optional[str] = Field(None, description="Token content for 'token' type")
    sources: Optional[List[SourceInfo]] = Field(None, description="Sources for 'sources' type")
    retrieval_info: Optional[RetrievalInfo] = Field(None, description="Retrieval info for 'sources' type")
    retrieval_metadata: Optional[Dict[str, Any]] = Field(None, description="Confidence and quality metrics for 'sources' type")
    complete_response: Optional[QueryResponse] = Field(None, description="Complete response for 'complete' type")
    error: Optional[str] = Field(None, description="Error message if any")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str = Field(..., description="Overall health status")
    components: ComponentHealth = Field(..., description="Component health details")
    collection_info: CollectionInfo = Field(..., description="Collection information")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "components": {
                    "retriever": "healthy",
                    "generator": "healthy"
                },
                "collection_info": {
                    "total_documents": 1000,
                    "file_types": ["html", "pdf", "docx"],
                    "sample_sources": ["Overview Document", "Technical Guide"]
                }
            }
        }
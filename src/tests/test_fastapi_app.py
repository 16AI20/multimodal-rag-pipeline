"""
Comprehensive test suite for FastAPI RAG application endpoints.
Tests all API routes with proper mocking and error handling for production readiness.

This test suite covers:
- Health check endpoint (/health)
- Collection info endpoint (/collection)
- Query endpoint (/query) - both regular and streaming
- Retrieval endpoint (/retrieve)
- Root endpoint (/)
- Error handling and edge cases
- CORS middleware
- Request/response validation
- Authentication and rate limiting (if implemented)

Each endpoint is tested for:
- Successful responses with valid data
- Error handling for invalid requests
- Proper HTTP status codes
- Response schema validation
- Streaming functionality
- Concurrent request handling
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List

# Import the FastAPI app and dependencies
from src.interfaces.fastapi_app import app
from src.api.schemas import (
    QueryRequest, QueryResponse, RetrievalRequest, RetrievalResponse,
    HealthResponse, CollectionInfo, StreamChunk, SourceInfo
)


class TestFastAPIApp:
    """Test suite for FastAPI application endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock RAG pipeline for testing."""
        mock_pipeline = Mock()
        
        # Mock successful query response
        mock_pipeline.query.return_value = {
            'query': 'What is the Sample Program?',
            'answer': 'The Sample Educational Program is a comprehensive training program.',
            'sources': [
                {
                    'citation': 'Program Overview',
                    'file_type': 'html',
                    'content_preview': 'The Sample Educational Program is designed to...'
                }
            ],
            'num_sources': 1,
            'retrieval_info': {
                'num_documents': 1,
                'file_types': ['html'],
                'sources': ['Program Overview'],
                'file_type_counts': {'html': 1},
                'search_filter': None
            },
            'retrieval_metadata': {
                'confidence_score': 0.85,
                'retrieval_quality': 'good'
            }
        }
        
        # Mock successful retrieval response
        mock_pipeline.retrieve_only.return_value = {
            'query': 'machine learning',
            'documents': [
                {
                    'rank': 1,
                    'source': 'ML Guide',
                    'file_type': 'pdf',
                    'content_preview': 'Machine learning is a subset of artificial intelligence...',
                    'similarity_score': 0.92
                }
            ],
            'num_documents': 1,
            'retrieval_info': {
                'num_documents': 1,
                'file_types': ['pdf'],
                'sources': ['ML Guide'],
                'file_type_counts': {'pdf': 1}
            }
        }
        
        # Mock collection info
        mock_pipeline.get_collection_info.return_value = {
            'total_documents': 1000,
            'file_types': ['html', 'pdf', 'docx'],
            'file_type_counts': {'html': 500, 'pdf': 300, 'docx': 200},
            'sample_sources': ['Program Overview', 'ML Guide', 'AI Handbook']
        }
        
        # Mock health check
        mock_pipeline.health_check.return_value = {
            'status': 'healthy',
            'components': {
                'retriever': 'healthy',
                'generator': 'healthy'
            },
            'collection_info': {
                'total_documents': 1000,
                'file_types': ['html', 'pdf', 'docx'],
                'file_type_counts': {'html': 500, 'pdf': 300, 'docx': 200},
                'sample_sources': ['Program Overview', 'ML Guide']
            }
        }
        
        # Mock streaming generator
        async def mock_streaming_generator():
            for token in ["The ", "Sample ", "Educational ", "Program ", "is ", "great."]:
                yield token
        
        mock_pipeline.generate_streaming_answer = AsyncMock(side_effect=lambda *args, **kwargs: mock_streaming_generator())
        
        return mock_pipeline
    
    @pytest.fixture(autouse=True)
    def setup_rag_pipeline(self, mock_rag_pipeline):
        """Automatically setup mock RAG pipeline for all tests."""
        with patch('src.interfaces.fastapi_app.rag_pipeline', mock_rag_pipeline):
            yield mock_rag_pipeline


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client, mock_rag_pipeline):
        """Test successful health check."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'components' in data
        assert data['components']['retriever'] == 'healthy'
        assert data['components']['generator'] == 'healthy'
        assert 'collection_info' in data
        
        # Verify RAG pipeline was called with correct parameters
        mock_rag_pipeline.health_check.assert_called_once_with(skip_generator_test=True)
    
    def test_health_check_pipeline_not_initialized(self, client):
        """Test health check when RAG pipeline is not initialized."""
        with patch('src.interfaces.fastapi_app.rag_pipeline', None):
            response = client.get("/health")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "RAG pipeline not initialized" in response.json()['detail']
    
    def test_health_check_pipeline_error(self, client, mock_rag_pipeline):
        """Test health check when pipeline throws error."""
        mock_rag_pipeline.health_check.side_effect = Exception("Health check failed")
        
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Health check failed" in response.json()['detail']
    
    def test_health_check_response_schema(self, client, mock_rag_pipeline):
        """Test health check response matches schema."""
        response = client.get("/health")
        data = response.json()
        
        # Validate against Pydantic schema
        health_response = HealthResponse(**data)
        assert health_response.status == 'healthy'
        assert health_response.components.retriever == 'healthy'
        assert health_response.components.generator == 'healthy'
        assert health_response.collection_info.total_documents == 1000


class TestCollectionEndpoint:
    """Test collection info endpoint."""
    
    def test_get_collection_info_success(self, client, mock_rag_pipeline):
        """Test successful collection info retrieval."""
        response = client.get("/collection")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data['total_documents'] == 1000
        assert set(data['file_types']) == {'html', 'pdf', 'docx'}
        assert data['file_type_counts']['html'] == 500
        assert len(data['sample_sources']) > 0
        
        mock_rag_pipeline.get_collection_info.assert_called_once()
    
    def test_get_collection_info_pipeline_not_initialized(self, client):
        """Test collection info when pipeline not initialized."""
        with patch('src.interfaces.fastapi_app.rag_pipeline', None):
            response = client.get("/collection")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_get_collection_info_error(self, client, mock_rag_pipeline):
        """Test collection info with error from pipeline."""
        mock_rag_pipeline.get_collection_info.return_value = {
            'error': 'Database connection failed'
        }
        
        response = client.get("/collection")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'Database connection failed' in response.json()['detail']
    
    def test_get_collection_info_exception(self, client, mock_rag_pipeline):
        """Test collection info when pipeline throws exception."""
        mock_rag_pipeline.get_collection_info.side_effect = Exception("Unexpected error")
        
        response = client.get("/collection")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'Unexpected error' in response.json()['detail']
    
    def test_collection_info_response_schema(self, client, mock_rag_pipeline):
        """Test collection info response schema validation."""
        response = client.get("/collection")
        data = response.json()
        
        # Validate against Pydantic schema
        collection_info = CollectionInfo(**data)
        assert collection_info.total_documents == 1000
        assert len(collection_info.file_types) == 3
        assert isinstance(collection_info.file_type_counts, dict)


class TestQueryEndpoint:
    """Test query endpoint for RAG queries."""
    
    def test_query_success(self, client, mock_rag_pipeline):
        """Test successful query processing."""
        request_data = {
            "question": "What is the Sample Program?",
            "k": 5,
            "temperature": 0.7,
            "return_sources": True,
            "stream": False
        }
        
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data['query'] == "What is the Sample Program?"
        assert 'Sample Educational Program' in data['answer']
        assert len(data['sources']) == 1
        assert data['num_sources'] == 1
        assert 'retrieval_info' in data
        assert 'retrieval_metadata' in data
        
        # Verify pipeline was called correctly
        mock_rag_pipeline.query.assert_called_once_with(
            question="What is the Sample Program?",
            k=5,
            file_type=None,
            return_sources=True
        )
    
    def test_query_with_file_type_filter(self, client, mock_rag_pipeline):
        """Test query with file type filter."""
        request_data = {
            "question": "Machine learning techniques",
            "k": 3,
            "file_type": "pdf",
            "stream": False
        }
        
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        mock_rag_pipeline.query.assert_called_once_with(
            question="Machine learning techniques",
            k=3,
            file_type="pdf",
            return_sources=True
        )
    
    def test_query_pipeline_not_initialized(self, client):
        """Test query when pipeline not initialized."""
        with patch('src.interfaces.fastapi_app.rag_pipeline', None):
            request_data = {"question": "Test question", "stream": False}
            response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_query_pipeline_error(self, client, mock_rag_pipeline):
        """Test query when pipeline returns error."""
        mock_rag_pipeline.query.return_value = {
            'error': 'LLM service unavailable',
            'query': 'Test question'
        }
        
        request_data = {"question": "Test question", "stream": False}
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'LLM service unavailable' in response.json()['detail']
    
    def test_query_none_answer(self, client, mock_rag_pipeline):
        """Test query when pipeline returns None answer."""
        mock_rag_pipeline.query.return_value = {
            'query': 'Test question',
            'answer': None,
            'sources': []
        }
        
        request_data = {"question": "Test question", "stream": False}
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'None answer' in response.json()['detail']
    
    def test_query_invalid_request(self, client):
        """Test query with invalid request data."""
        # Empty question
        request_data = {"question": "", "stream": False}
        response = client.post("/query", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Missing question
        request_data = {"k": 5, "stream": False}
        response = client.post("/query", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid k value
        request_data = {"question": "Test", "k": 0, "stream": False}
        response = client.post("/query", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid temperature
        request_data = {"question": "Test", "temperature": -1.0, "stream": False}
        response = client.post("/query", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_response_schema(self, client, mock_rag_pipeline):
        """Test query response schema validation."""
        request_data = {"question": "What is the Sample Program?", "stream": False}
        response = client.post("/query", json=request_data)
        data = response.json()
        
        # Validate against Pydantic schema
        query_response = QueryResponse(**data)
        assert query_response.query == "What is the Sample Program?"
        assert len(query_response.sources) == 1
        assert query_response.num_sources == 1


class TestQueryStreamingEndpoint:
    """Test streaming query functionality."""
    
    def test_query_streaming_success(self, client, mock_rag_pipeline):
        """Test successful streaming query."""
        request_data = {
            "question": "What is the Sample Program?",
            "k": 5,
            "stream": True
        }
        
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'
        
        # Parse SSE stream
        content = response.text
        assert 'data: ' in content
        
        # Should contain sources chunk and token chunks
        lines = content.strip().split('\n')
        data_lines = [line for line in lines if line.startswith('data: ')]
        
        assert len(data_lines) > 0
        
        # Parse first data chunk (should be sources)
        first_chunk_data = json.loads(data_lines[0][6:])  # Remove 'data: '
        assert first_chunk_data['type'] == 'sources'
        assert 'sources' in first_chunk_data
    
    def test_query_streaming_pipeline_error(self, client, mock_rag_pipeline):
        """Test streaming query with pipeline error."""
        mock_rag_pipeline.retrieve_only.return_value = {
            'error': 'Retrieval failed'
        }
        
        request_data = {
            "question": "Test question",
            "stream": True
        }
        
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        content = response.text
        assert 'error' in content
        assert 'Retrieval failed' in content
    
    def test_query_streaming_pipeline_not_initialized(self, client):
        """Test streaming query when pipeline not initialized."""
        with patch('src.interfaces.fastapi_app.rag_pipeline', None):
            request_data = {
                "question": "Test question",
                "stream": True
            }
            
            response = client.post("/query", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            
            content = response.text
            assert 'RAG pipeline not initialized' in content
    
    @pytest.mark.asyncio
    async def test_streaming_generation_error(self, client, mock_rag_pipeline):
        """Test streaming with generation error."""
        # Mock streaming generator that raises an error
        async def error_generator():
            raise Exception("Streaming generation failed")
        
        mock_rag_pipeline.generate_streaming_answer = AsyncMock(side_effect=error_generator)
        
        request_data = {
            "question": "Test question",
            "stream": True
        }
        
        response = client.post("/query", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        content = response.text
        assert 'error' in content.lower()


class TestRetrieveEndpoint:
    """Test retrieval-only endpoint."""
    
    def test_retrieve_success(self, client, mock_rag_pipeline):
        """Test successful document retrieval."""
        request_data = {
            "question": "machine learning",
            "k": 5,
            "include_scores": True
        }
        
        response = client.post("/retrieve", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data['query'] == "machine learning"
        assert data['num_documents'] == 1
        assert len(data['documents']) == 1
        assert 'retrieval_info' in data
        
        # Check document structure
        doc = data['documents'][0]
        assert doc['rank'] == 1
        assert doc['source'] == 'ML Guide'
        assert doc['file_type'] == 'pdf'
        assert 'similarity_score' in doc
        
        mock_rag_pipeline.retrieve_only.assert_called_once_with(
            question="machine learning",
            k=5,
            file_type=None,
            include_scores=True
        )
    
    def test_retrieve_with_file_type(self, client, mock_rag_pipeline):
        """Test retrieval with file type filter."""
        request_data = {
            "question": "AI concepts",
            "k": 3,
            "file_type": "html",
            "include_scores": False
        }
        
        response = client.post("/retrieve", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        mock_rag_pipeline.retrieve_only.assert_called_once_with(
            question="AI concepts",
            k=3,
            file_type="html",
            include_scores=False
        )
    
    def test_retrieve_pipeline_not_initialized(self, client):
        """Test retrieval when pipeline not initialized."""
        with patch('src.interfaces.fastapi_app.rag_pipeline', None):
            request_data = {"question": "Test question"}
            response = client.post("/retrieve", json=request_data)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_retrieve_pipeline_error(self, client, mock_rag_pipeline):
        """Test retrieval when pipeline returns error."""
        mock_rag_pipeline.retrieve_only.return_value = {
            'error': 'ChromaDB connection failed'
        }
        
        request_data = {"question": "Test question"}
        response = client.post("/retrieve", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'ChromaDB connection failed' in response.json()['detail']
    
    def test_retrieve_invalid_request(self, client):
        """Test retrieval with invalid request data."""
        # Empty question
        request_data = {"question": ""}
        response = client.post("/retrieve", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid k value
        request_data = {"question": "Test", "k": 25}  # Over limit
        response = client.post("/retrieve", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrieve_response_schema(self, client, mock_rag_pipeline):
        """Test retrieval response schema validation."""
        request_data = {"question": "machine learning"}
        response = client.post("/retrieve", json=request_data)
        data = response.json()
        
        # Validate against Pydantic schema
        retrieval_response = RetrievalResponse(**data)
        assert retrieval_response.query == "machine learning"
        assert retrieval_response.num_documents == 1
        assert len(retrieval_response.documents) == 1


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data['name'] == 'RAG API'
        assert data['version'] == '1.0.0'
        assert 'endpoints' in data
        assert 'health' in data['endpoints']
        assert 'query' in data['endpoints']
        assert 'retrieve' in data['endpoints']
        assert 'collection' in data['endpoints']


class TestCORSMiddleware:
    """Test CORS middleware functionality."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.options("/health")
        
        # Check CORS headers
        assert 'access-control-allow-origin' in response.headers
        assert 'access-control-allow-methods' in response.headers
        assert 'access-control-allow-headers' in response.headers
    
    def test_cors_preflight_request(self, client):
        """Test CORS preflight request handling."""
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        response = client.options("/query", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers['access-control-allow-origin'] == 'http://localhost:3000'


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 for unknown endpoints."""
        response = client.get("/unknown")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_method_not_allowed(self, client):
        """Test method not allowed errors."""
        response = client.delete("/health")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_invalid_json_body(self, client):
        """Test invalid JSON in request body."""
        response = client.post(
            "/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_content_type(self, client):
        """Test request without proper content type."""
        response = client.post("/query", data='{"question": "test"}')
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestConcurrentRequests:
    """Test concurrent request handling."""
    
    def test_concurrent_health_checks(self, client, mock_rag_pipeline):
        """Test multiple concurrent health check requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status_code == 200 for status_code in results)
        assert len(results) == 5
    
    def test_concurrent_queries(self, client, mock_rag_pipeline):
        """Test multiple concurrent query requests."""
        import threading
        
        results = []
        
        def make_query():
            request_data = {
                "question": "What is the Sample Program?",
                "stream": False
            }
            response = client.post("/query", json=request_data)
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(3):  # Fewer threads for heavier operations
            thread = threading.Thread(target=make_query)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status_code == 200 for status_code in results)
        assert len(results) == 3


class TestRequestValidation:
    """Test comprehensive request validation."""
    
    def test_query_request_validation(self, client):
        """Test query request validation edge cases."""
        # Test boundary values for k
        valid_request = {"question": "test", "k": 1, "stream": False}
        response = client.post("/query", json=valid_request)
        assert response.status_code == status.HTTP_200_OK
        
        valid_request = {"question": "test", "k": 20, "stream": False}
        response = client.post("/query", json=valid_request)
        assert response.status_code == status.HTTP_200_OK
        
        # Test boundary values for temperature
        valid_request = {"question": "test", "temperature": 0.0, "stream": False}
        response = client.post("/query", json=valid_request)
        assert response.status_code == status.HTTP_200_OK
        
        valid_request = {"question": "test", "temperature": 2.0, "stream": False}
        response = client.post("/query", json=valid_request)
        assert response.status_code == status.HTTP_200_OK
        
        # Test invalid values
        invalid_request = {"question": "test", "k": 21, "stream": False}  # Over limit
        response = client.post("/query", json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        invalid_request = {"question": "test", "temperature": 2.1, "stream": False}  # Over limit
        response = client.post("/query", json=invalid_request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrieval_request_validation(self, client):
        """Test retrieval request validation edge cases."""
        # Valid file types
        valid_request = {"question": "test", "file_type": "pdf"}
        response = client.post("/retrieve", json=valid_request)
        assert response.status_code == status.HTTP_200_OK
        
        # Empty file type should be treated as None
        valid_request = {"question": "test", "file_type": ""}
        response = client.post("/retrieve", json=valid_request)
        assert response.status_code == status.HTTP_200_OK
    
    def test_question_length_validation(self, client):
        """Test question length validation."""
        # Very long question (should still work)
        long_question = "What is " + "very " * 100 + "long question?"
        request_data = {"question": long_question, "stream": False}
        response = client.post("/query", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        # Single character question
        request_data = {"question": "?", "stream": False}
        response = client.post("/query", json=request_data)
        assert response.status_code == status.HTTP_200_OK


class TestResponseSchemas:
    """Test response schema compliance."""
    
    def test_all_response_schemas_valid(self, client, mock_rag_pipeline):
        """Test that all endpoints return valid response schemas."""
        # Health endpoint
        response = client.get("/health")
        HealthResponse(**response.json())
        
        # Collection endpoint
        response = client.get("/collection")
        CollectionInfo(**response.json())
        
        # Query endpoint
        request_data = {"question": "test", "stream": False}
        response = client.post("/query", json=request_data)
        QueryResponse(**response.json())
        
        # Retrieve endpoint
        request_data = {"question": "test"}
        response = client.post("/retrieve", json=request_data)
        RetrievalResponse(**response.json())
    
    def test_error_response_schemas(self, client):
        """Test error response schemas."""
        # Test 422 validation error
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422
        error_data = response.json()
        assert 'detail' in error_data
        
        # Test 404 error
        response = client.get("/nonexistent")
        assert response.status_code == 404
        error_data = response.json()
        assert 'detail' in error_data


class TestPerformance:
    """Test performance characteristics."""
    
    def test_health_check_performance(self, client, mock_rag_pipeline):
        """Test health check response time."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 1.0  # Should complete within 1 second
    
    def test_collection_info_performance(self, client, mock_rag_pipeline):
        """Test collection info response time."""
        import time
        
        start_time = time.time()
        response = client.get("/collection")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 2.0  # Should complete within 2 seconds
    
    def test_query_performance(self, client, mock_rag_pipeline):
        """Test query response time with mocked pipeline."""
        import time
        
        request_data = {"question": "What is the Sample Program?", "stream": False}
        
        start_time = time.time()
        response = client.post("/query", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 5.0  # Should complete within 5 seconds with mocking
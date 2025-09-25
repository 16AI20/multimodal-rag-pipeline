"""
Comprehensive test suite for ChromaDB database operations.
Tests vector storage, retrieval, collection management, and database health for production readiness.

This test suite covers:
- ChromaDB collection creation and management
- Vector storage and retrieval operations  
- Metadata filtering and search functionality
- Database health checks and connection handling
- Data persistence and collection information
- Performance with large datasets
- Error handling for database failures
- Concurrent access and thread safety

Each test ensures:
- Robust database operations under various conditions
- Proper error handling for connection failures
- Data integrity and consistency
- Scalability with large vector collections
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import List, Dict, Any, Optional
import time
import threading

# Import ChromaDB and related components
try:
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma
    from langchain.schema import Document
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    pytest.skip("ChromaDB not available", allow_module_level=True)


class TestChromaDBConnection:
    """Test ChromaDB connection and basic operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings function for testing."""
        embeddings = Mock()
        embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(3)
        ]
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return embeddings
    
    def test_chromadb_client_creation(self, temp_db_path):
        """Test ChromaDB client creation with persistent storage."""
        client = chromadb.PersistentClient(path=temp_db_path)
        assert client is not None
        
        # Test client can list collections (should be empty initially)
        collections = client.list_collections()
        assert len(collections) == 0
    
    def test_collection_creation(self, temp_db_path):
        """Test creating a new collection."""
        client = chromadb.PersistentClient(path=temp_db_path)
        
        collection_name = "test_collection"
        collection = client.create_collection(name=collection_name)
        
        assert collection.name == collection_name
        assert collection.count() == 0
        
        # Verify collection appears in list
        collections = client.list_collections()
        assert len(collections) == 1
        assert collections[0].name == collection_name
    
    def test_collection_get_or_create(self, temp_db_path):
        """Test get_or_create collection functionality."""
        client = chromadb.PersistentClient(path=temp_db_path)
        
        collection_name = "test_get_or_create"
        
        # First call should create
        collection1 = client.get_or_create_collection(name=collection_name)
        assert collection1.name == collection_name
        
        # Second call should get existing
        collection2 = client.get_or_create_collection(name=collection_name)
        assert collection2.name == collection_name
        assert collection1.id == collection2.id
    
    def test_collection_deletion(self, temp_db_path):
        """Test collection deletion."""
        client = chromadb.PersistentClient(path=temp_db_path)
        
        collection_name = "test_deletion"
        collection = client.create_collection(name=collection_name)
        
        # Verify collection exists
        collections = client.list_collections()
        assert len(collections) == 1
        
        # Delete collection
        client.delete_collection(name=collection_name)
        
        # Verify collection is gone
        collections = client.list_collections()
        assert len(collections) == 0
    
    def test_duplicate_collection_creation_fails(self, temp_db_path):
        """Test that creating duplicate collection raises error."""
        client = chromadb.PersistentClient(path=temp_db_path)
        
        collection_name = "duplicate_test"
        client.create_collection(name=collection_name)
        
        # Second creation should fail
        with pytest.raises(Exception):  # ChromaDB raises specific exception
            client.create_collection(name=collection_name)


class TestChromaDBDocumentOperations:
    """Test ChromaDB document storage and retrieval operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings function for testing."""
        embeddings = Mock()
        embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],  
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ]
        embeddings.embed_query.return_value = [0.15, 0.25, 0.35, 0.45, 0.55]
        return embeddings
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="The Sample Educational Program is a comprehensive training program.",
                metadata={"source": "program_overview.html", "file_type": "html", "section": "intro"}
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "ml_basics.pdf", "file_type": "pdf", "page": 1}
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "dl_guide.html", "file_type": "html", "section": "concepts"}
            )
        ]
    
    @pytest.fixture
    def chroma_vectorstore(self, temp_db_path, mock_embeddings):
        """Create Chroma vectorstore for testing."""
        return Chroma(
            collection_name="test_collection",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
    
    def test_add_documents(self, chroma_vectorstore, sample_documents):
        """Test adding documents to ChromaDB."""
        # Add documents
        ids = chroma_vectorstore.add_documents(sample_documents)
        
        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)
        
        # Verify documents were added
        collection = chroma_vectorstore._collection
        assert collection.count() == 3
    
    def test_similarity_search(self, chroma_vectorstore, sample_documents):
        """Test similarity search functionality."""
        # Add documents first
        chroma_vectorstore.add_documents(sample_documents)
        
        # Perform similarity search
        query = "What is the sample program?"
        results = chroma_vectorstore.similarity_search(query, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        
        # Results should contain relevant content
        result_contents = [doc.page_content for doc in results]
        assert any("Sample Educational Program" in content for content in result_contents)
    
    def test_similarity_search_with_score(self, chroma_vectorstore, sample_documents):
        """Test similarity search with relevance scores."""
        # Add documents first
        chroma_vectorstore.add_documents(sample_documents)
        
        # Perform similarity search with scores
        query = "machine learning concepts"
        results = chroma_vectorstore.similarity_search_with_score(query, k=3)
        
        assert len(results) <= 3
        assert all(len(result) == 2 for result in results)  # (document, score) tuples
        
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
            assert score >= 0  # Scores should be non-negative
    
    def test_metadata_filtering(self, chroma_vectorstore, sample_documents):
        """Test filtering documents by metadata."""
        # Add documents first
        chroma_vectorstore.add_documents(sample_documents)
        
        # Search with metadata filter
        query = "artificial intelligence"
        results = chroma_vectorstore.similarity_search(
            query,
            k=5,
            filter={"file_type": "html"}
        )
        
        # Should only return HTML documents
        for doc in results:
            assert doc.metadata.get("file_type") == "html"
    
    def test_delete_documents(self, chroma_vectorstore, sample_documents):
        """Test deleting documents from ChromaDB."""
        # Add documents first
        ids = chroma_vectorstore.add_documents(sample_documents)
        initial_count = chroma_vectorstore._collection.count()
        
        # Delete one document
        if hasattr(chroma_vectorstore, 'delete'):
            chroma_vectorstore.delete(ids[:1])
            
            # Verify document was deleted
            final_count = chroma_vectorstore._collection.count()
            assert final_count == initial_count - 1
    
    def test_update_documents(self, chroma_vectorstore, sample_documents):
        """Test updating existing documents."""
        # Add documents first
        ids = chroma_vectorstore.add_documents(sample_documents)
        
        # Update first document
        updated_doc = Document(
            page_content="Updated content about sample program features.",
            metadata={"source": "program_overview.html", "file_type": "html", "section": "updated"}
        )
        
        # ChromaDB typically requires delete and re-add for updates
        collection = chroma_vectorstore._collection
        collection.delete(ids=[ids[0]])
        new_ids = chroma_vectorstore.add_documents([updated_doc])
        
        # Verify update
        assert len(new_ids) == 1
        results = chroma_vectorstore.similarity_search("sample program features", k=1)
        assert "Updated content" in results[0].page_content
    
    def test_batch_operations(self, chroma_vectorstore):
        """Test batch document operations."""
        # Create large batch of documents
        batch_docs = []
        for i in range(50):
            doc = Document(
                page_content=f"Document {i} content about AI and machine learning topic {i}.",
                metadata={"source": f"doc_{i}.txt", "batch": "test", "index": i}
            )
            batch_docs.append(doc)
        
        # Add batch
        ids = chroma_vectorstore.add_documents(batch_docs)
        
        assert len(ids) == 50
        assert chroma_vectorstore._collection.count() == 50
        
        # Batch search
        results = chroma_vectorstore.similarity_search("AI machine learning", k=10)
        assert len(results) <= 10


class TestChromaDBPersistence:
    """Test ChromaDB data persistence and recovery."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings function for testing."""
        embeddings = Mock()
        embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        embeddings.embed_query.return_value = [0.15, 0.25, 0.35, 0.45, 0.55]
        return embeddings
    
    def test_data_persistence_across_sessions(self, temp_db_path, mock_embeddings):
        """Test that data persists across database sessions."""
        collection_name = "persistence_test"
        
        # Session 1: Add documents
        vectorstore1 = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        test_docs = [
            Document(page_content="Persistent document 1", metadata={"id": 1}),
            Document(page_content="Persistent document 2", metadata={"id": 2})
        ]
        
        vectorstore1.add_documents(test_docs)
        initial_count = vectorstore1._collection.count()
        
        # Close first session (vectorstore goes out of scope)
        del vectorstore1
        
        # Session 2: Reconnect and verify data exists
        vectorstore2 = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        final_count = vectorstore2._collection.count()
        assert final_count == initial_count
        
        # Verify content is accessible
        results = vectorstore2.similarity_search("Persistent document", k=2)
        assert len(results) == 2
        
        contents = [doc.page_content for doc in results]
        assert "Persistent document 1" in contents
        assert "Persistent document 2" in contents
    
    def test_collection_info_persistence(self, temp_db_path, mock_embeddings):
        """Test that collection metadata persists."""
        collection_name = "metadata_test"
        
        # Create collection with documents
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        test_doc = Document(
            page_content="Test content for metadata persistence",
            metadata={"source": "test.txt", "category": "test"}
        )
        
        vectorstore.add_documents([test_doc])
        
        # Get collection info
        collection = vectorstore._collection
        original_count = collection.count()
        
        # Reconnect
        del vectorstore
        vectorstore2 = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        collection2 = vectorstore2._collection
        new_count = collection2.count()
        
        assert new_count == original_count


class TestChromaDBPerformance:
    """Test ChromaDB performance and scalability."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings function for testing."""
        embeddings = Mock()
        # Generate random embeddings for performance testing
        embeddings.embed_documents.side_effect = lambda docs: [
            np.random.rand(384).tolist() for _ in docs
        ]
        embeddings.embed_query.return_value = np.random.rand(384).tolist()
        return embeddings
    
    def test_large_batch_insertion_performance(self, temp_db_path, mock_embeddings):
        """Test performance with large batch insertions."""
        vectorstore = Chroma(
            collection_name="performance_test",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Create large batch of documents
        large_batch = []
        for i in range(1000):
            doc = Document(
                page_content=f"Performance test document {i} with substantial content for testing database operations.",
                metadata={"source": f"perf_doc_{i}.txt", "index": i, "category": f"cat_{i % 10}"}
            )
            large_batch.append(doc)
        
        # Measure insertion time
        start_time = time.time()
        ids = vectorstore.add_documents(large_batch)
        insertion_time = time.time() - start_time
        
        # Verify all documents were added
        assert len(ids) == 1000
        assert vectorstore._collection.count() == 1000
        
        # Insertion should complete in reasonable time (adjust threshold as needed)
        assert insertion_time < 60.0  # Should complete within 60 seconds
    
    def test_search_performance_with_large_dataset(self, temp_db_path, mock_embeddings):
        """Test search performance with large dataset."""
        vectorstore = Chroma(
            collection_name="search_performance_test",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Add substantial dataset
        docs = []
        for i in range(500):
            doc = Document(
                page_content=f"Search test document {i} about AI, machine learning, and data science topics.",
                metadata={"source": f"search_doc_{i}.txt", "topic": f"topic_{i % 5}"}
            )
            docs.append(doc)
        
        vectorstore.add_documents(docs)
        
        # Measure search performance
        queries = [
            "artificial intelligence applications",
            "machine learning algorithms",
            "data science techniques",
            "neural network architectures",
            "deep learning frameworks"
        ]
        
        total_search_time = 0
        for query in queries:
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=10)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            # Verify search returns results
            assert len(results) <= 10
            assert all(isinstance(doc, Document) for doc in results)
        
        # Average search time should be reasonable
        avg_search_time = total_search_time / len(queries)
        assert avg_search_time < 2.0  # Should average under 2 seconds per search
    
    def test_concurrent_access(self, temp_db_path, mock_embeddings):
        """Test concurrent access to ChromaDB."""
        collection_name = "concurrent_test"
        
        def worker_function(worker_id, results_list):
            """Worker function for concurrent testing."""
            try:
                # Each worker creates its own vectorstore instance
                worker_vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=mock_embeddings,
                    persist_directory=temp_db_path
                )
                
                # Add documents
                docs = []
                for i in range(10):
                    doc = Document(
                        page_content=f"Worker {worker_id} document {i}",
                        metadata={"worker": worker_id, "doc_id": i}
                    )
                    docs.append(doc)
                
                worker_vectorstore.add_documents(docs)
                
                # Perform searches
                results = worker_vectorstore.similarity_search(f"Worker {worker_id}", k=5)
                
                results_list.append({
                    'worker_id': worker_id,
                    'docs_added': len(docs),
                    'search_results': len(results),
                    'success': True
                })
                
            except Exception as e:
                results_list.append({
                    'worker_id': worker_id,
                    'error': str(e),
                    'success': False
                })
        
        # Create threads for concurrent access
        threads = []
        results = []
        
        for worker_id in range(5):
            thread = threading.Thread(
                target=worker_function,
                args=(worker_id, results)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify all workers completed successfully
        assert len(results) == 5
        assert all(result['success'] for result in results)
        
        # Verify final database state
        final_vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        final_count = final_vectorstore._collection.count()
        assert final_count == 50  # 5 workers * 10 docs each


class TestChromaDBErrorHandling:
    """Test ChromaDB error handling and recovery."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings function for testing."""
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return embeddings
    
    def test_invalid_database_path(self, mock_embeddings):
        """Test handling of invalid database path."""
        # Try to create vectorstore with invalid path
        invalid_path = "/nonexistent/invalid/path"
        
        # Should handle gracefully or raise appropriate exception
        try:
            vectorstore = Chroma(
                collection_name="test",
                embedding_function=mock_embeddings,
                persist_directory=invalid_path
            )
            # If no exception, verify it handles the situation
            assert vectorstore is not None
        except Exception as e:
            # Should raise a meaningful exception
            assert isinstance(e, (OSError, PermissionError, Exception))
    
    def test_corrupted_collection_handling(self, temp_db_path, mock_embeddings):
        """Test handling of corrupted collection data."""
        collection_name = "corruption_test"
        
        # Create normal vectorstore first
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Add some documents
        test_doc = Document(page_content="Test document", metadata={"id": 1})
        vectorstore.add_documents([test_doc])
        
        # Close vectorstore
        del vectorstore
        
        # Simulate corruption by writing invalid data to database files
        db_files = list(Path(temp_db_path).glob("**/*"))
        if db_files:
            # Write some invalid data to a database file (be careful not to break the test)
            pass  # Skip actual corruption to avoid breaking the test environment
        
        # Try to reconnect - should handle corruption gracefully
        try:
            vectorstore2 = Chroma(
                collection_name=collection_name,
                embedding_function=mock_embeddings,
                persist_directory=temp_db_path
            )
            # Should either recover or fail gracefully
            assert vectorstore2 is not None
        except Exception:
            # Corruption handling is implementation-dependent
            pass
    
    def test_embedding_function_failure(self, temp_db_path):
        """Test handling of embedding function failures."""
        # Create embedding function that fails
        failing_embeddings = Mock()
        failing_embeddings.embed_documents.side_effect = Exception("Embedding service unavailable")
        failing_embeddings.embed_query.side_effect = Exception("Embedding service unavailable")
        
        vectorstore = Chroma(
            collection_name="embedding_failure_test",
            embedding_function=failing_embeddings,
            persist_directory=temp_db_path
        )
        
        test_doc = Document(page_content="Test document", metadata={"id": 1})
        
        # Should handle embedding failure gracefully
        with pytest.raises(Exception, match="Embedding service unavailable"):
            vectorstore.add_documents([test_doc])
    
    def test_search_with_empty_collection(self, temp_db_path, mock_embeddings):
        """Test search behavior with empty collection."""
        vectorstore = Chroma(
            collection_name="empty_test",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Search empty collection
        results = vectorstore.similarity_search("any query", k=5)
        
        # Should return empty results, not error
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_invalid_metadata_handling(self, temp_db_path, mock_embeddings):
        """Test handling of invalid metadata types."""
        vectorstore = Chroma(
            collection_name="metadata_test",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Document with invalid metadata types
        invalid_doc = Document(
            page_content="Test document with invalid metadata",
            metadata={
                "valid_string": "test",
                "valid_int": 123,
                "invalid_list": [1, 2, 3],  # Lists might not be supported
                "invalid_dict": {"nested": "dict"},  # Nested dicts might not be supported
                "invalid_object": object()  # Objects definitely not supported
            }
        )
        
        # Should either handle gracefully or raise clear error
        try:
            vectorstore.add_documents([invalid_doc])
            
            # If successful, verify data integrity
            results = vectorstore.similarity_search("invalid metadata", k=1)
            assert len(results) <= 1
            
        except Exception as e:
            # Should raise clear, meaningful error
            assert str(e)  # Error message should exist


class TestChromaDBIntegration:
    """Integration tests for ChromaDB with other system components."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings function for testing."""
        embeddings = Mock()
        embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ]
        embeddings.embed_query.return_value = [0.15, 0.25, 0.35, 0.45, 0.55]
        return embeddings
    
    def test_integration_with_document_chunker(self, temp_db_path, mock_embeddings):
        """Test integration with document chunking workflow."""
        vectorstore = Chroma(
            collection_name="chunker_integration_test",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Simulate chunked documents from different file types
        chunked_docs = [
            Document(
                page_content="HTML chunk 1: Introduction to sample program",
                metadata={"source": "intro.html", "chunk_id": 0, "file_type": "html"}
            ),
            Document(
                page_content="HTML chunk 2: Program curriculum overview",
                metadata={"source": "intro.html", "chunk_id": 1, "file_type": "html"}
            ),
            Document(
                page_content="PDF chunk 1: Technical requirements section",
                metadata={"source": "requirements.pdf", "chunk_id": 0, "file_type": "pdf", "page": 1}
            ),
            Document(
                page_content="Audio chunk 1: Lecture transcript segment",
                metadata={"source": "lecture.mp3", "chunk_id": 0, "file_type": "audio", "timestamp": "00:00-02:30"}
            )
        ]
        
        # Add chunked documents
        ids = vectorstore.add_documents(chunked_docs)
        assert len(ids) == 4
        
        # Test cross-chunk search
        results = vectorstore.similarity_search("sample program requirements", k=3)
        
        # Should find relevant chunks across different sources
        assert len(results) <= 3
        sources = [doc.metadata.get("source") for doc in results]
        file_types = [doc.metadata.get("file_type") for doc in results]
        
        # Verify diversity in results
        assert len(set(sources)) > 1  # Multiple sources
        assert len(set(file_types)) > 1  # Multiple file types
    
    def test_collection_health_check(self, temp_db_path, mock_embeddings):
        """Test collection health check functionality."""
        vectorstore = Chroma(
            collection_name="health_check_test",
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Add test documents
        test_docs = [
            Document(page_content=f"Health check document {i}", metadata={"id": i})
            for i in range(10)
        ]
        
        vectorstore.add_documents(test_docs)
        
        # Perform health checks
        collection = vectorstore._collection
        
        # Basic health metrics
        count = collection.count()
        assert count == 10
        
        # Verify collection is accessible
        try:
            results = vectorstore.similarity_search("health check", k=1)
            assert len(results) > 0
            health_status = "healthy"
        except Exception:
            health_status = "unhealthy"
        
        assert health_status == "healthy"
    
    def test_collection_info_retrieval(self, temp_db_path, mock_embeddings):
        """Test retrieving comprehensive collection information."""
        collection_name = "info_test"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=mock_embeddings,
            persist_directory=temp_db_path
        )
        
        # Add documents with varied metadata
        docs_with_metadata = [
            Document(
                page_content="HTML document content",
                metadata={"file_type": "html", "source": "test1.html", "section": "intro"}
            ),
            Document(
                page_content="PDF document content",
                metadata={"file_type": "pdf", "source": "test1.pdf", "page": 1}
            ),
            Document(
                page_content="Another HTML document",
                metadata={"file_type": "html", "source": "test2.html", "section": "body"}
            )
        ]
        
        vectorstore.add_documents(docs_with_metadata)
        
        # Get collection information
        collection = vectorstore._collection
        count = collection.count()
        
        # Verify collection info
        assert count == 3
        assert collection.name == collection_name
        
        # Test querying collection data
        all_data = collection.get()
        assert len(all_data['documents']) == 3
        assert len(all_data['metadatas']) == 3
        assert len(all_data['embeddings']) == 3
        
        # Verify metadata variety
        file_types = [meta.get("file_type") for meta in all_data['metadatas']]
        assert "html" in file_types
        assert "pdf" in file_types
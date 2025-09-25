"""
Comprehensive test suite for DocumentRetriever class.
Tests all retrieval functionality with mocked ChromaDB and embedding models for production readiness.

This test suite covers:
- DocumentRetriever initialization and configuration
- Similarity search with various query types and parameters
- Document filtering by file type and metadata
- Reranking with cross-encoder models (when available)
- Confidence score calculation and retrieval quality assessment
- Retrieval metadata generation and source summarization
- Error handling for ChromaDB failures and missing models
- Performance testing with large result sets
- Integration with different embedding models

Each test mocks external dependencies (ChromaDB, HuggingFace models) to ensure:
- Fast test execution without external service dependencies
- Consistent test results regardless of external service availability
- Comprehensive error condition testing
- Isolation from network and hardware variations
"""

import pytest
import os
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import List, Dict, Any, Tuple
import numpy as np

from src.rag.retriever import DocumentRetriever
from langchain.schema import Document


class TestDocumentRetrieverInitialization:
    """Test DocumentRetriever initialization and configuration."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for retriever."""
        return {
            'embeddings': {
                'model': 'BAAI/bge-large-en-v1.5',
                'device': 'cpu',
                'reranking_enabled': False,
                'reranking_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'reranking_top_k': 10
            },
            'vectordb': {
                'path': 'test_vector_store',
                'collection_name': 'test_collection'
            }
        }
    
    @pytest.fixture
    def mock_chroma_collection(self):
        """Mock ChromaDB collection."""
        mock_collection = Mock()
        mock_collection.count.return_value = 1000
        mock_collection.get.return_value = {
            'metadatas': [
                {'file_type': 'html', 'citation_source': 'test.html'},
                {'file_type': 'pdf', 'citation_source': 'test.pdf'},
                {'file_type': 'html', 'citation_source': 'another.html'}
            ]
        }
        return mock_collection
    
    @pytest.fixture
    def mock_vectorstore(self, mock_chroma_collection):
        """Mock ChromaDB vectorstore."""
        mock_store = Mock()
        mock_store._collection = mock_chroma_collection
        mock_store.as_retriever.return_value = Mock()
        return mock_store
    
    @pytest.fixture
    def retriever(self, mock_config, mock_vectorstore):
        """Create DocumentRetriever with mocked dependencies."""
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('src.rag.retriever.Chroma', return_value=mock_vectorstore):
            
            mock_embeddings.return_value = Mock()
            retriever = DocumentRetriever()
            return retriever
    
    def test_retriever_initialization_success(self, retriever):
        """Test successful retriever initialization."""
        assert retriever.db_path == 'test_vector_store'
        assert retriever.collection_name == 'test_collection'
        assert retriever.default_k == 5
        assert retriever.embeddings is not None
        assert retriever.vectorstore is not None
        assert retriever.retriever is not None
        assert retriever.reranking_enabled is False
    
    def test_retriever_initialization_with_overrides(self, mock_config):
        """Test retriever initialization with parameter overrides."""
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma'):
            
            retriever = DocumentRetriever(
                db_path="custom_db",
                collection_name="custom_collection",
                embedding_model="custom/model",
                default_k=10
            )
            
            assert retriever.db_path == "custom_db"
            assert retriever.collection_name == "custom_collection"
            assert retriever.default_k == 10
    
    def test_retriever_initialization_with_reranking(self, mock_config):
        """Test retriever initialization with reranking enabled."""
        mock_config['embeddings']['reranking_enabled'] = True
        
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma'), \
             patch('sentence_transformers.CrossEncoder') as mock_cross_encoder:
            
            mock_cross_encoder.return_value = Mock()
            
            retriever = DocumentRetriever()
            
            assert retriever.reranking_enabled is True
            assert retriever.reranking_model is not None
            mock_cross_encoder.assert_called_once()
    
    def test_retriever_initialization_reranking_import_error(self, mock_config):
        """Test retriever initialization when sentence-transformers not available."""
        mock_config['embeddings']['reranking_enabled'] = True
        
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma'), \
             patch('builtins.__import__', side_effect=ImportError("No module named 'sentence_transformers'")):
            
            retriever = DocumentRetriever()
            
            # Should gracefully disable reranking
            assert retriever.reranking_enabled is False
            assert retriever.reranking_model is None
    
    def test_retriever_initialization_with_container_environment(self, mock_config):
        """Test retriever initialization in container environment."""
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma'), \
             patch.dict(os.environ, {'FORCE_CPU_DEVICE': 'true'}):
            
            retriever = DocumentRetriever()
            
            # Should force CPU device in container environment
            assert retriever is not None


class TestDocumentRetrieval:
    """Test document retrieval functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="The Sample Educational Program is a comprehensive training program.",
                metadata={
                    'citation_source': 'Program Overview',
                    'file_type': 'html',
                    'source': 'sample_overview.html'
                }
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence.",
                metadata={
                    'citation_source': 'ML Basics',
                    'file_type': 'pdf',
                    'source': 'ml_basics.pdf'
                }
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers.",
                metadata={
                    'citation_source': 'Deep Learning Guide',
                    'file_type': 'pdf',
                    'source': 'deep_learning.pdf'
                }
            )
        ]
    
    @pytest.fixture
    def retriever_with_docs(self, retriever, sample_documents):
        """Retriever configured with sample documents."""
        # Mock similarity search to return sample documents
        retriever.vectorstore.similarity_search.return_value = sample_documents
        retriever.vectorstore.similarity_search_with_score.return_value = [
            (doc, 0.1 + i * 0.1) for i, doc in enumerate(sample_documents)
        ]
        return retriever
    
    def test_retrieve_documents_success(self, retriever_with_docs):
        """Test successful document retrieval."""
        with patch('src.rag.retriever.preprocess_query', return_value="test query"):
            documents = retriever_with_docs.retrieve_documents("test query", k=3)
        
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].metadata['citation_source'] == 'Program Overview'
        retriever_with_docs.vectorstore.similarity_search.assert_called_once()
    
    def test_retrieve_documents_with_preprocessing(self, retriever_with_docs):
        """Test document retrieval with query preprocessing."""
        with patch('src.rag.retriever.preprocess_query', return_value="processed query") as mock_preprocess:
            retriever_with_docs.retrieve_documents("original query", k=2)
        
        mock_preprocess.assert_called_once_with("original query", config_path="conf/config.yaml")
        retriever_with_docs.vectorstore.similarity_search.assert_called_with("processed query", k=2)
    
    def test_retrieve_documents_error_handling(self, retriever):
        """Test document retrieval error handling."""
        retriever.vectorstore.similarity_search.side_effect = Exception("ChromaDB error")
        
        documents = retriever.retrieve_documents("test query")
        
        assert len(documents) == 0
    
    def test_retrieve_with_scores_success(self, retriever_with_docs):
        """Test retrieval with similarity scores."""
        with patch('src.rag.retriever.preprocess_query', return_value="test query"):
            docs_with_scores = retriever_with_docs.retrieve_with_scores("test query", k=2)
        
        assert len(docs_with_scores) == 3  # Returns all mock documents
        assert all(isinstance(item, tuple) and len(item) == 2 for item in docs_with_scores)
        
        doc, score = docs_with_scores[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_retrieve_with_scores_and_reranking(self, mock_config, mock_vectorstore, sample_documents):
        """Test retrieval with reranking enabled."""
        mock_config['embeddings']['reranking_enabled'] = True
        
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma', return_value=mock_vectorstore), \
             patch('sentence_transformers.CrossEncoder') as mock_cross_encoder:
            
            # Mock cross-encoder reranking scores
            mock_cross_encoder.return_value.predict.return_value = [0.9, 0.7, 0.8]
            
            retriever = DocumentRetriever()
            retriever.vectorstore.similarity_search_with_score.return_value = [
                (doc, 0.1 + i * 0.1) for i, doc in enumerate(sample_documents)
            ]
            
            with patch('src.rag.retriever.preprocess_query', return_value="test query"):
                docs_with_scores = retriever.retrieve_with_scores("test query", k=3)
            
            # Should have been reranked
            assert len(docs_with_scores) == 3
            # First document should have highest reranking score (0.9)
            assert docs_with_scores[0][1] == 0.9
    
    def test_retrieve_by_file_type_success(self, retriever_with_docs):
        """Test retrieval filtered by file type."""
        # Mock filtered search
        pdf_docs = [doc for doc in retriever_with_docs.vectorstore.similarity_search.return_value 
                   if doc.metadata.get('file_type') == 'pdf']
        retriever_with_docs.vectorstore.similarity_search.return_value = pdf_docs
        
        documents = retriever_with_docs.retrieve_by_file_type("machine learning", "pdf", k=5)
        
        assert all(doc.metadata.get('file_type') == 'pdf' for doc in documents)
        retriever_with_docs.vectorstore.similarity_search.assert_called_with(
            "machine learning", k=10, filter={"file_type": "pdf"}
        )
    
    def test_retrieve_by_file_type_error(self, retriever):
        """Test file type retrieval error handling."""
        retriever.vectorstore.similarity_search.side_effect = Exception("Filter error")
        
        documents = retriever.retrieve_by_file_type("test", "pdf")
        
        assert len(documents) == 0
    
    def test_retrieve_documents_default_k(self, retriever_with_docs):
        """Test retrieval with default k value."""
        with patch('src.rag.retriever.preprocess_query', return_value="test"):
            retriever_with_docs.retrieve_documents("test")
        
        retriever_with_docs.vectorstore.similarity_search.assert_called_with("test", k=5)


class TestCollectionInfo:
    """Test collection information functionality."""
    
    def test_get_collection_info_success(self, retriever, mock_chroma_collection):
        """Test successful collection info retrieval."""
        info = retriever.get_collection_info()
        
        assert info['total_documents'] == 1000
        assert 'html' in info['file_types']
        assert 'pdf' in info['file_types']
        assert info['file_type_counts']['html'] == 2
        assert info['file_type_counts']['pdf'] == 1
        assert len(info['sample_sources']) > 0
    
    def test_get_collection_info_empty_collection(self, retriever, mock_chroma_collection):
        """Test collection info for empty collection."""
        mock_chroma_collection.count.return_value = 0
        
        info = retriever.get_collection_info()
        
        assert info['total_documents'] == 0
        assert info['file_types'] == []
        assert info['file_type_counts'] == {}
        assert info['sample_sources'] == []
    
    def test_get_collection_info_error(self, retriever, mock_chroma_collection):
        """Test collection info error handling."""
        mock_chroma_collection.count.side_effect = Exception("Collection error")
        
        info = retriever.get_collection_info()
        
        assert 'error' in info
        assert 'Collection error' in info['error']


class TestSourceSummary:
    """Test source summary functionality."""
    
    def test_get_source_summary_success(self, retriever, sample_documents):
        """Test successful source summary generation."""
        summary = retriever.get_source_summary(sample_documents)
        
        assert summary['num_documents'] == 3
        assert set(summary['file_types']) == {'html', 'pdf'}
        assert set(summary['sources']) == {'Program Overview', 'ML Basics', 'Deep Learning Guide'}
        assert summary['file_type_counts']['html'] == 1
        assert summary['file_type_counts']['pdf'] == 2
    
    def test_get_source_summary_empty_documents(self, retriever):
        """Test source summary with empty document list."""
        summary = retriever.get_source_summary([])
        
        assert summary['num_documents'] == 0
        assert summary['file_types'] == []
        assert summary['sources'] == []
    
    def test_get_source_summary_missing_metadata(self, retriever):
        """Test source summary with documents missing metadata."""
        docs_with_missing_metadata = [
            Document(page_content="Test content", metadata={}),
            Document(page_content="More content", metadata={'file_type': 'txt'})
        ]
        
        summary = retriever.get_source_summary(docs_with_missing_metadata)
        
        assert summary['num_documents'] == 2
        assert 'unknown' in summary['file_types']
        assert 'txt' in summary['file_types']
        assert 'unknown' in summary['sources']


class TestRetrievalConfidence:
    """Test retrieval confidence calculation."""
    
    def test_calculate_retrieval_confidence_high_quality(self, retriever):
        """Test confidence calculation for high-quality results."""
        # Mock high-quality results (low distance scores)
        docs_with_scores = [
            (Document(page_content="Relevant content", metadata={'citation_source': 'source1', 'file_type': 'html'}), 0.1),
            (Document(page_content="Also relevant", metadata={'citation_source': 'source2', 'file_type': 'pdf'}), 0.15),
            (Document(page_content="Very relevant", metadata={'citation_source': 'source3', 'file_type': 'html'}), 0.2)
        ]
        
        confidence = retriever.calculate_retrieval_confidence(docs_with_scores, "test query")
        
        assert confidence['confidence_score'] > 0.8  # High confidence
        assert confidence['confidence_level'] == 'high'
        assert confidence['retrieval_quality'] == 'excellent'
        assert confidence['num_documents_found'] == 3
        assert confidence['diverse_sources'] is True
        assert confidence['unique_sources'] == 3
        assert len(confidence['file_types_found']) == 2
    
    def test_calculate_retrieval_confidence_low_quality(self, retriever):
        """Test confidence calculation for low-quality results."""
        # Mock low-quality results (high distance scores)
        docs_with_scores = [
            (Document(page_content="Barely relevant", metadata={'citation_source': 'source1', 'file_type': 'html'}), 1.8),
            (Document(page_content="Not very relevant", metadata={'citation_source': 'source1', 'file_type': 'html'}), 1.9)
        ]
        
        confidence = retriever.calculate_retrieval_confidence(docs_with_scores, "test query")
        
        assert confidence['confidence_score'] < 0.4  # Low confidence
        assert confidence['confidence_level'] == 'very_low'
        assert confidence['retrieval_quality'] == 'poor'
        assert confidence['diverse_sources'] is False  # Same source
        assert confidence['unique_sources'] == 1
    
    def test_calculate_retrieval_confidence_empty_results(self, retriever):
        """Test confidence calculation with no results."""
        confidence = retriever.calculate_retrieval_confidence([], "test query")
        
        assert confidence['confidence_score'] == 0.0
        assert confidence['confidence_level'] == 'very_low'
        assert confidence['retrieval_quality'] == 'poor'
        assert confidence['num_documents_found'] == 0
        assert confidence['best_match_score'] == 0.0
        assert confidence['diverse_sources'] is False
    
    def test_calculate_retrieval_confidence_medium_quality(self, retriever):
        """Test confidence calculation for medium-quality results."""
        docs_with_scores = [
            (Document(page_content="Somewhat relevant", metadata={'citation_source': 'source1', 'file_type': 'pdf'}), 0.6),
            (Document(page_content="Moderately relevant", metadata={'citation_source': 'source2', 'file_type': 'html'}), 0.8)
        ]
        
        confidence = retriever.calculate_retrieval_confidence(docs_with_scores, "test query")
        
        assert 0.4 <= confidence['confidence_score'] < 0.8
        assert confidence['confidence_level'] in ['low', 'medium']
        assert confidence['retrieval_quality'] in ['fair', 'good']
        assert len(confidence['score_distribution']) == 2


class TestReranking:
    """Test document reranking functionality."""
    
    def test_rerank_documents_success(self, mock_config, mock_vectorstore):
        """Test successful document reranking."""
        mock_config['embeddings']['reranking_enabled'] = True
        
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma', return_value=mock_vectorstore), \
             patch('sentence_transformers.CrossEncoder') as mock_cross_encoder:
            
            # Mock reranking scores (higher is better for cross-encoder)
            mock_cross_encoder.return_value.predict.return_value = [0.9, 0.3, 0.7]
            
            retriever = DocumentRetriever()
            
            docs_with_scores = [
                (Document(page_content="Doc 1", metadata={}), 0.1),
                (Document(page_content="Doc 2", metadata={}), 0.2),
                (Document(page_content="Doc 3", metadata={}), 0.3)
            ]
            
            reranked = retriever._rerank_documents("test query", docs_with_scores, top_k=3)
            
            # Should be reordered by reranking scores: Doc 1 (0.9), Doc 3 (0.7), Doc 2 (0.3)
            assert len(reranked) == 3
            assert reranked[0][1] == 0.9  # Highest reranking score first
            assert reranked[1][1] == 0.7
            assert reranked[2][1] == 0.3
    
    def test_rerank_documents_disabled(self, retriever):
        """Test reranking when disabled."""
        docs_with_scores = [
            (Document(page_content="Doc 1", metadata={}), 0.1),
            (Document(page_content="Doc 2", metadata={}), 0.2)
        ]
        
        # Reranking is disabled by default in our fixture
        reranked = retriever._rerank_documents("test query", docs_with_scores)
        
        # Should return unchanged
        assert reranked == docs_with_scores
    
    def test_rerank_documents_empty_list(self, retriever):
        """Test reranking with empty document list."""
        reranked = retriever._rerank_documents("test query", [])
        
        assert reranked == []
    
    def test_rerank_documents_error_handling(self, mock_config, mock_vectorstore):
        """Test reranking error handling."""
        mock_config['embeddings']['reranking_enabled'] = True
        
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma', return_value=mock_vectorstore), \
             patch('sentence_transformers.CrossEncoder') as mock_cross_encoder:
            
            # Mock reranking error
            mock_cross_encoder.return_value.predict.side_effect = Exception("Reranking failed")
            
            retriever = DocumentRetriever()
            
            docs_with_scores = [
                (Document(page_content="Doc 1", metadata={}), 0.1)
            ]
            
            # Should fallback to original scores on error
            reranked = retriever._rerank_documents("test query", docs_with_scores)
            assert reranked == docs_with_scores


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_initialization_with_invalid_config(self):
        """Test retriever initialization with invalid configuration."""
        with patch('src.rag.retriever.load_config', side_effect=Exception("Config error")):
            with pytest.raises(Exception, match="Config error"):
                DocumentRetriever()
    
    def test_embeddings_initialization_error(self, mock_config):
        """Test handling of embeddings initialization error."""
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings', side_effect=Exception("Model loading failed")):
            
            with pytest.raises(Exception, match="Model loading failed"):
                DocumentRetriever()
    
    def test_chroma_initialization_error(self, mock_config):
        """Test handling of ChromaDB initialization error."""
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma', side_effect=Exception("ChromaDB error")):
            
            with pytest.raises(Exception, match="ChromaDB error"):
                DocumentRetriever()
    
    def test_retrieval_with_corrupted_metadata(self, retriever):
        """Test retrieval handling corrupted document metadata."""
        # Mock documents with corrupted metadata
        corrupted_docs = [
            Document(page_content="Content", metadata=None),  # None metadata
            Document(page_content="Content", metadata={'file_type': None}),  # None values
        ]
        
        retriever.vectorstore.similarity_search.return_value = corrupted_docs
        
        with patch('src.rag.retriever.preprocess_query', return_value="test"):
            documents = retriever.retrieve_documents("test")
        
        # Should handle gracefully and return documents
        assert len(documents) == 2
        
        # Test source summary with corrupted metadata
        summary = retriever.get_source_summary(corrupted_docs)
        assert summary['num_documents'] == 2
        assert 'unknown' in summary['file_types']


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_result_set_handling(self, retriever):
        """Test handling of large result sets."""
        # Mock large number of documents
        large_doc_set = [
            Document(
                page_content=f"Document {i} content",
                metadata={
                    'citation_source': f'source_{i}',
                    'file_type': 'pdf' if i % 2 == 0 else 'html'
                }
            ) for i in range(1000)
        ]
        
        retriever.vectorstore.similarity_search.return_value = large_doc_set
        
        with patch('src.rag.retriever.preprocess_query', return_value="test"):
            documents = retriever.retrieve_documents("test", k=1000)
        
        assert len(documents) == 1000
        
        # Test source summary performance
        import time
        start_time = time.time()
        summary = retriever.get_source_summary(documents)
        end_time = time.time()
        
        assert end_time - start_time < 2.0  # Should complete within 2 seconds
        assert summary['num_documents'] == 1000
    
    def test_confidence_calculation_performance(self, retriever):
        """Test confidence calculation performance with many documents."""
        # Create large set of documents with scores
        large_docs_with_scores = [
            (
                Document(
                    page_content=f"Content {i}",
                    metadata={'citation_source': f'source_{i}', 'file_type': 'pdf'}
                ),
                0.1 + (i * 0.001)  # Varying scores
            ) for i in range(500)
        ]
        
        import time
        start_time = time.time()
        confidence = retriever.calculate_retrieval_confidence(large_docs_with_scores, "test query")
        end_time = time.time()
        
        assert end_time - start_time < 1.0  # Should complete within 1 second
        assert confidence['num_documents_found'] == 500
        assert len(confidence['score_distribution']) == 500


class TestIntegration:
    """Integration tests for retriever functionality."""
    
    def test_end_to_end_retrieval_flow(self, mock_config, mock_vectorstore, sample_documents):
        """Test complete retrieval flow."""
        with patch('src.rag.retriever.load_config', return_value=mock_config), \
             patch('src.rag.retriever.HuggingFaceEmbeddings'), \
             patch('src.rag.retriever.Chroma', return_value=mock_vectorstore):
            
            retriever = DocumentRetriever()
            
            # Mock retrieval results
            retriever.vectorstore.similarity_search_with_score.return_value = [
                (doc, 0.1 + i * 0.1) for i, doc in enumerate(sample_documents)
            ]
            
            with patch('src.rag.retriever.preprocess_query', return_value="AI apprenticeship"):
                # Retrieve with scores
                docs_with_scores = retriever.retrieve_with_scores("AI apprenticeship", k=3)
                
                # Calculate confidence
                confidence = retriever.calculate_retrieval_confidence(docs_with_scores, "AI apprenticeship")
                
                # Get source summary
                documents = [doc for doc, _ in docs_with_scores]
                summary = retriever.get_source_summary(documents)
            
            # Verify complete flow
            assert len(docs_with_scores) == 3
            assert confidence['confidence_score'] > 0.8  # High quality results
            assert summary['num_documents'] == 3
            assert len(summary['file_types']) == 2  # html and pdf
    
    def test_retrieval_with_different_query_types(self, retriever_with_docs):
        """Test retrieval with various query types and lengths."""
        test_queries = [
            "AI",  # Short query
            "What is artificial intelligence?",  # Question
            "machine learning deep learning neural networks",  # Keywords
            "Explain the concept of " + "supervised learning " * 20,  # Very long query
            "AI/ML программирование",  # Mixed languages/special chars
            "",  # Empty query (should be handled gracefully)
        ]
        
        for query in test_queries:
            with patch('src.rag.retriever.preprocess_query', return_value=query or "processed"):
                try:
                    documents = retriever_with_docs.retrieve_documents(query, k=2)
                    # Should not crash and return documents for non-empty queries
                    if query.strip():
                        assert len(documents) <= 3  # Limited by mock data
                except Exception as e:
                    # Should not raise exceptions for any query type
                    pytest.fail(f"Query '{query}' caused exception: {e}")
    
    def test_retrieval_consistency(self, retriever_with_docs):
        """Test that repeated retrieval calls return consistent results."""
        query = "test query"
        
        with patch('src.rag.retriever.preprocess_query', return_value=query):
            # Make multiple retrieval calls
            results = []
            for _ in range(5):
                docs = retriever_with_docs.retrieve_documents(query, k=3)
                results.append([doc.metadata.get('citation_source') for doc in docs])
            
            # Results should be consistent
            first_result = results[0]
            for result in results[1:]:
                assert result == first_result
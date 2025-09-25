"""
Unit tests for RAG pipeline query processing functionality.
Tests the high-priority RAG pipeline query processing features.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.rag.pipeline import RAGPipeline


class TestRAGPipelineQueryProcessing:
    """Test RAG pipeline query processing functionality."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock document retriever."""
        retriever = MagicMock()
        retriever.similarity_search.return_value = [
            {
                'content': 'The Sample Educational Program is a comprehensive training program...',
                'metadata': {'source': 'sample_overview.html', 'score': 0.95}
            },
            {
                'content': 'The program covers machine learning, deep learning, and data science...',
                'metadata': {'source': 'curriculum.pdf', 'score': 0.87}
            }
        ]
        retriever.get_retrieval_metadata.return_value = {
            'confidence_score': 0.91,
            'retrieval_quality': 'high',
            'unique_sources': 2,
            'best_match_score': 0.95
        }
        return retriever
    
    @pytest.fixture
    def mock_generator(self):
        """Mock answer generator."""
        generator = MagicMock()
        generator.generate_answer.return_value = {
            'answer': 'The Sample Educational Program is a comprehensive training program designed to equip participants with practical AI and machine learning skills.',
            'success': True
        }
        return generator
    
    @pytest.fixture
    def rag_pipeline(self, mock_retriever, mock_generator, temp_config_file):
        """Create RAG pipeline with mocked components."""
        with patch('src.rag.pipeline.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.pipeline.AnswerGenerator', return_value=mock_generator):
            pipeline = RAGPipeline(config_path=temp_config_file)
            return pipeline
    
    def test_query_end_to_end_success(self, rag_pipeline, mock_retriever, mock_generator):
        """Test successful end-to-end query processing."""
        question = "What is the Sample Educational Program?"
        
        result = rag_pipeline.query(question, k=5)
        
        # Verify retriever was called correctly
        mock_retriever.similarity_search.assert_called_once_with(question, k=5, file_type=None)
        mock_retriever.get_retrieval_metadata.assert_called_once()
        
        # Verify generator was called with retrieved documents
        mock_generator.generate_answer.assert_called_once()
        
        # Verify response structure
        assert result['success'] is True
        assert result['query'] == question
        assert 'Sample Educational Program' in result['answer']
        assert len(result['sources']) == 2
        assert result['retrieval_metadata']['confidence_score'] == 0.91
        assert result['num_sources'] == 2
    
    def test_query_with_file_type_filter(self, rag_pipeline, mock_retriever, mock_generator):
        """Test query processing with file type filter."""
        question = "What programming languages are covered?"
        file_type = "pdf"
        
        result = rag_pipeline.query(question, k=3, file_type=file_type)
        
        # Verify retriever was called with file type filter
        mock_retriever.similarity_search.assert_called_once_with(question, k=3, file_type=file_type)
        
        assert result['success'] is True
        assert result['query'] == question
    
    def test_query_retrieval_failure(self, rag_pipeline, mock_retriever, mock_generator):
        """Test query processing when retrieval fails."""
        question = "What causes retrieval to fail?"
        
        # Mock retrieval failure
        mock_retriever.similarity_search.side_effect = Exception("ChromaDB connection error")
        
        result = rag_pipeline.query(question)
        
        # Should handle retrieval failure gracefully
        assert result['success'] is False
        assert 'error' in result
        assert 'ChromaDB connection error' in result['error']
        assert result['query'] == question
    
    def test_query_generation_failure(self, rag_pipeline, mock_retriever, mock_generator):
        """Test query processing when answer generation fails."""
        question = "What causes generation to fail?"
        
        # Mock generation failure
        mock_generator.generate_answer.return_value = {
            'answer': None,
            'success': False,
            'error': 'LLM service unavailable'
        }
        
        result = rag_pipeline.query(question)
        
        # Should handle generation failure gracefully
        assert result['success'] is False
        assert 'error' in result
        assert 'LLM service unavailable' in result['error']
        assert len(result['sources']) == 2  # Sources should still be available
    
    def test_query_no_documents_found(self, rag_pipeline, mock_retriever, mock_generator):
        """Test query processing when no relevant documents are found."""
        question = "Completely unrelated query about cooking"
        
        # Mock no documents found
        mock_retriever.similarity_search.return_value = []
        mock_retriever.get_retrieval_metadata.return_value = {
            'confidence_score': 0.1,
            'retrieval_quality': 'low',
            'unique_sources': 0,
            'best_match_score': 0.0
        }
        
        result = rag_pipeline.query(question)
        
        # Should handle no documents gracefully
        assert result['success'] is True  # Still successful, just no good matches
        assert len(result['sources']) == 0
        assert result['retrieval_metadata']['confidence_score'] == 0.1
    
    def test_query_parameter_validation(self, rag_pipeline):
        """Test query parameter validation."""
        # Test empty question
        result = rag_pipeline.query("")
        assert result['success'] is False
        assert 'error' in result
        
        # Test None question
        result = rag_pipeline.query(None)
        assert result['success'] is False
        assert 'error' in result
        
        # Test invalid k parameter
        result = rag_pipeline.query("Valid question", k=0)
        assert result['success'] is False
        assert 'error' in result
    
    def test_query_source_formatting(self, rag_pipeline, mock_retriever, mock_generator):
        """Test that query results format sources correctly."""
        question = "How are sources formatted?"
        
        result = rag_pipeline.query(question)
        
        # Verify source formatting
        assert len(result['sources']) == 2
        
        for source in result['sources']:
            assert 'citation' in source
            assert 'file_type' in source
            assert 'content_preview' in source
            # Should extract file type from source path
            if 'html' in source['citation']:
                assert source['file_type'] == 'html'
            elif 'pdf' in source['citation']:
                assert source['file_type'] == 'pdf'
    
    def test_query_retrieval_info_structure(self, rag_pipeline, mock_retriever, mock_generator):
        """Test that retrieval info is properly structured."""
        question = "Test retrieval info structure"
        
        result = rag_pipeline.query(question)
        
        # Verify retrieval_info structure
        assert 'retrieval_info' in result
        retrieval_info = result['retrieval_info']
        
        assert 'num_documents' in retrieval_info
        assert 'file_types' in retrieval_info
        assert 'sources' in retrieval_info
        assert 'file_type_counts' in retrieval_info
        
        # Verify calculated values
        assert retrieval_info['num_documents'] == 2
        assert 'html' in retrieval_info['file_types']
        assert 'pdf' in retrieval_info['file_types']
    
    def test_query_logging_and_monitoring(self, rag_pipeline, mock_retriever, mock_generator):
        """Test that queries are properly logged for monitoring."""
        question = "Test logging functionality"
        
        with patch('logging.getLogger') as mock_logger:
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance
            
            result = rag_pipeline.query(question)
            
            # Should log query processing steps
            assert logger_instance.info.called
            # Verify that query details are logged
            log_calls = [call.args[0] for call in logger_instance.info.call_args_list]
            assert any('query' in call.lower() for call in log_calls)
    
    def test_health_check(self, rag_pipeline, mock_retriever, mock_generator):
        """Test RAG pipeline health check functionality."""
        # Mock healthy components
        mock_retriever.health_check.return_value = {'status': 'healthy'}
        mock_generator.health_check.return_value = {'status': 'healthy'}
        
        health = rag_pipeline.health_check()
        
        assert health['status'] == 'healthy'
        assert health['components']['retriever'] == 'healthy'
        assert health['components']['generator'] == 'healthy'
    
    def test_health_check_unhealthy_components(self, rag_pipeline, mock_retriever, mock_generator):
        """Test health check with unhealthy components."""
        # Mock unhealthy retriever
        mock_retriever.health_check.return_value = {'status': 'unhealthy', 'error': 'DB connection failed'}
        mock_generator.health_check.return_value = {'status': 'healthy'}
        
        health = rag_pipeline.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['components']['retriever'] == 'unhealthy'
        assert health['components']['generator'] == 'healthy'
        assert 'error' in health
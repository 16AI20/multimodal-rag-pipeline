"""
Comprehensive edge case tests for RAG pipeline components.

Tests various failure modes, edge cases, and error conditions to ensure
robust system behavior under adverse conditions.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.rag.retriever import DocumentRetriever
from src.rag.generator import AnswerGenerator
from src.rag.pipeline import RAGPipeline
from src.interfaces.base_interfaces import RetrievalError, GenerationError
from src.utils.content_safety import check_input_safety, check_output_safety
from langchain.schema import Document


class TestEmptyInputs:
    """Test handling of empty and malformed inputs."""
    
    def test_empty_query_retrieval(self):
        """Test retrieval with empty query string."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            
            with pytest.raises(ValueError, match="Query must be a non-empty string"):
                retriever.retrieve_documents("")
    
    def test_whitespace_only_query(self):
        """Test retrieval with whitespace-only query."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            
            with pytest.raises(ValueError, match="Query must be a non-empty string"):
                retriever.retrieve_documents("   \n\t   ")
    
    def test_none_query(self):
        """Test retrieval with None query."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            
            with pytest.raises(ValueError):
                retriever.retrieve_documents(None)
    
    def test_zero_k_value(self):
        """Test retrieval with k=0."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            
            with pytest.raises(ValueError, match="k must be a positive integer"):
                retriever.retrieve_documents("test query", k=0)
    
    def test_negative_k_value(self):
        """Test retrieval with negative k value."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            
            with pytest.raises(ValueError, match="k must be a positive integer"):
                retriever.retrieve_documents("test query", k=-5)


class TestMalformedDocuments:
    """Test handling of malformed and problematic documents."""
    
    def test_empty_document_content(self):
        """Test generation with empty document content."""
        with patch('src.rag.generator.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            generator = AnswerGenerator()
            generator.llm = Mock()
            generator.prompt_template = Mock()
            generator.prompt_template.format.return_value = "test prompt"
            
            # Mock successful LLM response
            mock_response = Mock()
            mock_response.content = "Based on the provided context, I cannot find specific information to answer this question."
            generator.llm.invoke.return_value = mock_response
            
            empty_docs = [
                Document(page_content="", metadata={"source": "empty1.txt"}),
                Document(page_content="   ", metadata={"source": "empty2.txt"}),
                Document(page_content="\n\n\t", metadata={"source": "empty3.txt"})
            ]
            
            result = generator.generate_answer("test query", empty_docs)
            
            # Should handle gracefully without crashing
            assert "answer" in result
            assert result["sources"] == []  # No meaningful sources
    
    def test_documents_with_special_characters(self):
        """Test processing documents with special characters and encoding issues."""
        special_docs = [
            Document(page_content="Special chars: àáâãäåæçèéêë 中文 العربية 🚀🎯", metadata={"source": "unicode.txt"}),
            Document(page_content="Null bytes: \x00\x01\x02", metadata={"source": "binary.txt"}),
            Document(page_content="Very long line: " + "x" * 10000, metadata={"source": "long.txt"})
        ]
        
        with patch('src.rag.generator.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            generator = AnswerGenerator()
            generator.llm = Mock()
            generator.prompt_template = Mock()
            generator.prompt_template.format.return_value = "test prompt"
            
            mock_response = Mock()
            mock_response.content = "Test response"
            generator.llm.invoke.return_value = mock_response
            
            # Should handle special characters without crashing
            result = generator.generate_answer("test query", special_docs)
            assert "answer" in result
    
    def test_corrupted_metadata(self):
        """Test handling of documents with corrupted or missing metadata."""
        corrupted_docs = [
            Document(page_content="Content 1", metadata=None),
            Document(page_content="Content 2", metadata={}),
            Document(page_content="Content 3", metadata={"invalid_field": None}),
            Document(page_content="Content 4")  # No metadata attribute
        ]
        
        with patch('src.rag.generator.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            generator = AnswerGenerator()
            generator.llm = Mock()
            generator.prompt_template = Mock()
            generator.prompt_template.format.return_value = "test prompt"
            
            mock_response = Mock()
            mock_response.content = "Test response"
            generator.llm.invoke.return_value = mock_response
            
            # Should handle missing/corrupted metadata gracefully
            result = generator.generate_answer("test query", corrupted_docs)
            assert "answer" in result


class TestNetworkFailures:
    """Test handling of network and external service failures."""
    
    def test_ollama_connection_failure(self):
        """Test handling when Ollama service is unavailable."""
        with patch('src.rag.generator.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            with patch('src.rag.generator.ChatOllama') as mock_ollama:
                mock_ollama.side_effect = ConnectionError("Ollama service unavailable")
                
                with pytest.raises(RuntimeError, match="Ollama LLM initialization failed"):
                    AnswerGenerator()
    
    def test_huggingface_model_download_failure(self):
        """Test handling when HuggingFace model download fails."""
        config = self._get_mock_config()
        config.llm.provider = "huggingface"
        
        with patch('src.rag.generator.load_config') as mock_config:
            mock_config.return_value = config
            
            with patch('transformers.pipeline') as mock_pipeline:
                mock_pipeline.side_effect = OSError("Model download failed")
                
                # Should fallback to GPT-2
                generator = AnswerGenerator()
                assert generator.llm is not None  # Should have fallback model
    
    def test_vectorstore_connection_loss(self):
        """Test handling when vector database connection is lost."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            retriever.vectorstore.similarity_search.side_effect = ConnectionError("ChromaDB connection lost")
            
            with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
                retriever.retrieve_documents("test query")


class TestResourceLimits:
    """Test behavior under resource constraints and limits."""
    
    def test_extremely_large_query(self):
        """Test handling of very large query strings."""
        large_query = "test " * 10000  # Very large query
        
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            retriever.vectorstore.similarity_search.return_value = []
            
            # Should handle large queries without memory issues
            docs = retriever.retrieve_documents(large_query, k=5)
            assert isinstance(docs, list)
    
    def test_maximum_k_value(self):
        """Test retrieval with maximum allowed k value."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            retriever.vectorstore.similarity_search.return_value = []
            
            # Test with maximum reasonable k value
            docs = retriever.retrieve_documents("test query", k=50)
            assert isinstance(docs, list)
    
    def test_excessive_k_value(self):
        """Test retrieval with unreasonably large k value."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            
            with pytest.raises(ValueError, match="k must be an integer between 1 and 50"):
                retriever.retrieve_documents("test query", k=1000)


class TestConcurrencyIssues:
    """Test concurrent access and threading issues."""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling multiple concurrent queries."""
        import asyncio
        
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            retriever.vectorstore.similarity_search.return_value = [
                Document(page_content="Test content", metadata={"source": "test.txt"})
            ]
            
            # Create multiple concurrent queries
            queries = [f"test query {i}" for i in range(10)]
            
            # Execute queries concurrently
            tasks = [asyncio.create_task(asyncio.to_thread(retriever.retrieve_documents, query)) 
                    for query in queries]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All results should be successful (no exceptions)
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, list)


class TestContentSafetyEdgeCases:
    """Test content safety system with edge cases."""
    
    def test_mixed_safe_unsafe_content(self):
        """Test content with both safe and unsafe elements."""
        mixed_content = "How to learn machine learning and also how to hack systems?"
        
        safety_result = check_input_safety(mixed_content)
        assert not safety_result["is_safe"]
        assert "inappropriate_content" in safety_result.get("issue_type", "")
    
    def test_unicode_malicious_content(self):
        """Test malicious content with unicode characters."""
        unicode_content = "How to ḩáçk systems using üñîçøðé?"
        
        safety_result = check_input_safety(unicode_content)
        # Should detect harmful intent regardless of unicode
        assert not safety_result["is_safe"]
    
    def test_very_long_potentially_harmful_content(self):
        """Test very long content that might contain harmful elements."""
        long_content = "This is a very long question about learning " + "and studying " * 100 + " and how to hack systems"
        
        safety_result = check_input_safety(long_content)
        assert not safety_result["is_safe"]


class TestSystemIntegration:
    """Test integration between components under edge conditions."""
    
    def test_pipeline_with_all_empty_components(self):
        """Test RAG pipeline when all components return empty results."""
        with patch('src.rag.pipeline.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            pipeline = RAGPipeline()
            
            # Mock components to return empty results
            pipeline.retriever = Mock()
            pipeline.retriever.retrieve_documents.return_value = []
            
            pipeline.generator = Mock()
            pipeline.generator.generate_answer.return_value = {
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
            
            result = pipeline.query("test query")
            
            # Should handle gracefully
            assert "answer" in result
            assert "sources" in result
    
    def test_pipeline_with_partial_failures(self):
        """Test RAG pipeline when some components fail."""
        with patch('src.rag.pipeline.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            pipeline = RAGPipeline()
            
            # Mock retriever to succeed but generator to fail
            pipeline.retriever = Mock()
            pipeline.retriever.retrieve_documents.return_value = [
                Document(page_content="Test content", metadata={"source": "test.txt"})
            ]
            
            pipeline.generator = Mock()
            pipeline.generator.generate_answer.side_effect = GenerationError("LLM unavailable")
            
            with pytest.raises(GenerationError):
                pipeline.query("test query")


class TestConfigurationErrors:
    """Test various configuration error scenarios."""
    
    def test_missing_required_config_sections(self):
        """Test initialization with missing required configuration sections."""
        incomplete_config = {
            "embeddings": {"model": "test-model"},
            # Missing llm, vectordb sections
        }
        
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = incomplete_config
            
            # Should handle missing config gracefully with defaults
            retriever = DocumentRetriever()
            assert hasattr(retriever, 'config')
    
    def test_invalid_config_values(self):
        """Test initialization with invalid configuration values."""
        invalid_config = {
            "embeddings": {"model": ""},  # Empty model name
            "llm": {"provider": "invalid_provider"},  # Invalid provider
            "vectordb": {"collection_name": ""}  # Empty collection name
        }
        
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = invalid_config
            
            # Should handle invalid config and use defaults
            retriever = DocumentRetriever()
            assert hasattr(retriever, 'config')


class TestFileSystemErrors:
    """Test handling of file system related errors."""
    
    def test_nonexistent_vector_db_path(self):
        """Test initialization with non-existent vector database path."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            with patch('src.rag.retriever.Chroma') as mock_chroma:
                mock_chroma.side_effect = FileNotFoundError("Vector database not found")
                
                with pytest.raises(Exception):  # Should propagate initialization error
                    DocumentRetriever()
    
    def test_permission_denied_vector_db(self):
        """Test handling when vector database has permission issues."""
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            with patch('src.rag.retriever.Chroma') as mock_chroma:
                mock_chroma.side_effect = PermissionError("Permission denied")
                
                with pytest.raises(Exception):  # Should propagate permission error
                    DocumentRetriever()


class TestMemoryAndResourceLimits:
    """Test behavior under memory and resource constraints."""
    
    def test_large_number_of_documents(self):
        """Test retrieval with very large number of documents."""
        # Simulate large document set
        large_doc_set = [
            Document(page_content=f"Document {i} content", metadata={"source": f"doc_{i}.txt"})
            for i in range(1000)
        ]
        
        with patch('src.rag.retriever.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            retriever = DocumentRetriever()
            retriever.vectorstore = Mock()
            retriever.vectorstore.similarity_search.return_value = large_doc_set[:50]  # Return reasonable subset
            
            docs = retriever.retrieve_documents("test query", k=50)
            
            # Should handle large result sets efficiently
            assert len(docs) <= 50
            assert isinstance(docs, list)
    
    def test_very_long_document_content(self):
        """Test generation with extremely long document content."""
        very_long_content = "This is a very long document. " * 10000  # Very long content
        long_docs = [Document(page_content=very_long_content, metadata={"source": "long_doc.txt"})]
        
        with patch('src.rag.generator.load_config') as mock_config:
            mock_config.return_value = self._get_mock_config()
            
            generator = AnswerGenerator()
            generator.llm = Mock()
            generator.prompt_template = Mock()
            generator.prompt_template.format.return_value = "test prompt"
            
            mock_response = Mock()
            mock_response.content = "Test response"
            generator.llm.invoke.return_value = mock_response
            
            # Should handle long content without memory issues
            result = generator.generate_answer("test query", long_docs)
            assert "answer" in result


class TestRandomSeedConsistency:
    """Test random seed consistency for reproducible results."""
    
    def test_reproducible_results_with_seed(self):
        """Test that setting random seed produces consistent results."""
        from src.utils.core import set_random_seed
        
        # Set seed and run operation twice
        set_random_seed(42)
        result1 = self._run_mock_operation()
        
        set_random_seed(42)  # Reset to same seed
        result2 = self._run_mock_operation()
        
        # Results should be identical with same seed
        assert result1 == result2
    
    def _run_mock_operation(self):
        """Mock operation that uses randomness."""
        import random
        return [random.random() for _ in range(10)]


def _get_mock_config():
    """Helper to create mock configuration for testing."""
    return {
        "embeddings": {
            "model": "BAAI/bge-large-en-v1.5",
            "normalize_embeddings": True
        },
        "llm": {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "temperature": 0.7,
            "max_tokens": 512
        },
        "vectordb": {
            "path": "vector_store",
            "collection_name": "test_collection"
        },
        "prompts": {
            "rag_template": "Test template: {context}\n\nQuestion: {question}\n\nAnswer:"
        }
    }


# Fixtures for test setup
@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    return _get_mock_config()


@pytest.fixture
def temp_vector_db():
    """Create temporary vector database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        Document(page_content="Machine learning is a subset of AI.", metadata={"source": "ml_basics.txt", "file_type": "txt"}),
        Document(page_content="Deep learning uses neural networks.", metadata={"source": "dl_guide.pdf", "file_type": "pdf"}),
        Document(page_content="Natural language processing works with text.", metadata={"source": "nlp_intro.html", "file_type": "html"})
    ]


@pytest.fixture
def empty_documents():
    """Provide empty documents for edge case testing."""
    return [
        Document(page_content="", metadata={"source": "empty1.txt"}),
        Document(page_content="   ", metadata={"source": "empty2.txt"}),
        Document(page_content="\n\n", metadata={"source": "empty3.txt"})
    ]
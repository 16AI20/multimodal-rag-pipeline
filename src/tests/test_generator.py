"""
Comprehensive test suite for AnswerGenerator class.
Tests all LLM initialization and response generation with mocked services for production readiness.

This test suite covers:
- AnswerGenerator initialization with different LLM providers (Ollama, HuggingFace)
- Answer generation from documents with various scenarios
- Streaming response generation and async functionality
- Error handling for LLM service failures and network issues
- Prompt template handling and customization
- Context preparation from retrieved documents
- Response post-processing and cleaning
- Performance testing with large contexts
- Temperature and parameter handling

Each test mocks external dependencies (Ollama API, HuggingFace models) to ensure:
- Fast test execution without external LLM service dependencies
- Consistent test results regardless of LLM service availability
- Comprehensive error condition testing
- Isolation from network latency and API rate limits
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any, AsyncGenerator
import requests

from src.rag.generator import AnswerGenerator
from langchain.schema import Document
from langchain.prompts import PromptTemplate


class TestAnswerGeneratorInitialization:
    """Test AnswerGenerator initialization and configuration."""
    
    @pytest.fixture
    def mock_ollama_config(self):
        """Mock configuration for Ollama LLM."""
        return {
            'llm': {
                'provider': 'ollama',
                'model': 'llama3',
                'temperature': 0.7,
                'max_tokens': 512,
                'base_url': 'http://localhost:11434'
            },
            'prompts': {
                'rag_template': 'Context: {context}\nQuestion: {question}\nAnswer:',
                'input_variables': ['context', 'question']
            }
        }
    
    @pytest.fixture
    def mock_huggingface_config(self):
        """Mock configuration for HuggingFace LLM."""
        return {
            'llm': {
                'provider': 'huggingface',
                'model': 'microsoft/DialoGPT-medium',
                'temperature': 0.7,
                'max_tokens': 256
            },
            'prompts': {
                'rag_template': 'Based on: {context}\nQ: {question}\nA:',
                'input_variables': ['context', 'question']
            }
        }
    
    def test_generator_initialization_ollama_success(self, mock_ollama_config):
        """Test successful Ollama generator initialization."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            # Mock successful Ollama server response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'models': [{'name': 'llama3'}, {'name': 'mistral'}]
            }
            mock_get.return_value = mock_response
            
            mock_ollama.return_value = Mock()
            
            generator = AnswerGenerator()
            
            assert generator.provider == 'ollama'
            assert generator.llm_model == 'llama3'
            assert generator.temperature == 0.7
            assert generator.max_length == 512
            assert generator.base_url == 'http://localhost:11434'
            assert generator.llm is not None
            assert generator.prompt_template is not None
    
    def test_generator_initialization_ollama_server_not_running(self, mock_ollama_config):
        """Test Ollama initialization when server is not running."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get', side_effect=requests.exceptions.RequestException("Connection refused")):
            
            with pytest.raises(RuntimeError, match="Ollama LLM initialization failed"):
                AnswerGenerator()
    
    def test_generator_initialization_ollama_model_not_found(self, mock_ollama_config):
        """Test Ollama initialization when model is not available."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get:
            
            # Mock response with different models
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'models': [{'name': 'mistral'}, {'name': 'codellama'}]
            }
            mock_get.return_value = mock_response
            
            with pytest.raises(RuntimeError, match="Model 'llama3' not found in Ollama"):
                AnswerGenerator()
    
    def test_generator_initialization_huggingface_success(self, mock_huggingface_config):
        """Test successful HuggingFace generator initialization."""
        with patch('src.rag.generator.load_config', return_value=mock_huggingface_config), \
             patch('transformers.pipeline') as mock_pipeline, \
             patch('src.rag.generator.HuggingFacePipeline') as mock_hf_pipeline:
            
            mock_pipeline.return_value = Mock()
            mock_hf_pipeline.return_value = Mock()
            
            generator = AnswerGenerator()
            
            assert generator.provider == 'huggingface'
            assert generator.llm_model == 'microsoft/DialoGPT-medium'
            assert generator.llm is not None
            mock_pipeline.assert_called_once()
    
    def test_generator_initialization_huggingface_fallback_to_gpt2(self, mock_huggingface_config):
        """Test HuggingFace initialization with fallback to GPT-2."""
        with patch('src.rag.generator.load_config', return_value=mock_huggingface_config), \
             patch('transformers.pipeline') as mock_pipeline, \
             patch('src.rag.generator.HuggingFacePipeline') as mock_hf_pipeline:
            
            # First call fails (main model), second call succeeds (GPT-2)
            mock_pipeline.side_effect = [Exception("Model not found"), Mock()]
            mock_hf_pipeline.return_value = Mock()
            
            generator = AnswerGenerator()
            
            assert generator.llm is not None
            assert mock_pipeline.call_count == 2
            # Second call should be for GPT-2
            second_call_args = mock_pipeline.call_args_list[1]
            assert 'gpt2' in str(second_call_args)
    
    def test_generator_initialization_with_overrides(self, mock_ollama_config):
        """Test generator initialization with parameter overrides."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            # Mock successful Ollama response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'custom-model'}]}
            mock_get.return_value = mock_response
            mock_ollama.return_value = Mock()
            
            generator = AnswerGenerator(
                llm_model='custom-model',
                temperature=0.9,
                max_length=1024
            )
            
            assert generator.llm_model == 'custom-model'
            assert generator.temperature == 0.9
            assert generator.max_length == 1024
    
    def test_generator_prompt_template_creation(self, mock_ollama_config):
        """Test prompt template creation from configuration."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama3'}]}
            mock_get.return_value = mock_response
            mock_ollama.return_value = Mock()
            
            generator = AnswerGenerator()
            
            assert isinstance(generator.prompt_template, PromptTemplate)
            assert 'context' in generator.prompt_template.input_variables
            assert 'question' in generator.prompt_template.input_variables
            assert '{context}' in generator.prompt_template.template
            assert '{question}' in generator.prompt_template.template


class TestAnswerGeneration:
    """Test answer generation functionality."""
    
    @pytest.fixture
    def mock_generator_ollama(self, mock_ollama_config):
        """Create generator with mocked Ollama LLM."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama3'}]}
            mock_get.return_value = mock_response
            
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            
            generator = AnswerGenerator()
            return generator
    
    @pytest.fixture
    def mock_generator_huggingface(self, mock_huggingface_config):
        """Create generator with mocked HuggingFace LLM."""
        with patch('src.rag.generator.load_config', return_value=mock_huggingface_config), \
             patch('transformers.pipeline') as mock_pipeline, \
             patch('src.rag.generator.HuggingFacePipeline') as mock_hf_pipeline:
            
            mock_pipeline.return_value = Mock()
            mock_llm = Mock()
            mock_hf_pipeline.return_value = mock_llm
            
            generator = AnswerGenerator()
            return generator
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="The Sample Educational Program is a comprehensive 9-month training program that combines theoretical learning with hands-on project experience.",
                metadata={'citation_source': 'Program Overview', 'file_type': 'html'}
            ),
            Document(
                page_content="Participants in the program learn machine learning, deep learning, and data engineering skills through real-world industry projects.",
                metadata={'citation_source': 'Program Curriculum', 'file_type': 'pdf'}
            ),
            Document(
                page_content="The program includes mentorship from industry experts and career placement support upon completion.",
                metadata={'citation_source': 'Program Benefits', 'file_type': 'html'}
            )
        ]
    
    def test_generate_from_documents_ollama_success(self, mock_generator_ollama, sample_documents):
        """Test successful answer generation with Ollama."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.content = "The Sample Educational Program is a comprehensive 9-month training program designed to equip participants with practical AI and machine learning skills through hands-on projects and industry mentorship."
        mock_generator_ollama.llm.invoke.return_value = mock_response
        
        result = mock_generator_ollama.generate_from_documents(
            "What is the Sample Educational Program?",
            sample_documents
        )
        
        assert result['query'] == "What is the Sample Educational Program?"
        assert 'Sample Educational Program' in result['answer']
        assert len(result['sources']) == 3
        assert result['num_sources'] == 3
        assert all('citation' in source for source in result['sources'])
        assert all('file_type' in source for source in result['sources'])
        assert all('content_preview' in source for source in result['sources'])
    
    def test_generate_from_documents_huggingface_success(self, mock_generator_huggingface, sample_documents):
        """Test successful answer generation with HuggingFace."""
        # Mock HuggingFace response
        mock_generator_huggingface.llm.return_value = "Answer: The Sample Educational Program is a comprehensive training program for AI practitioners."
        
        result = mock_generator_huggingface.generate_from_documents(
            "What is the Sample Program?",
            sample_documents
        )
        
        assert result['query'] == "What is the Sample Program?"
        assert 'Sample Educational Program' in result['answer']
        assert len(result['sources']) == 3
        assert result['num_sources'] == 3
    
    def test_generate_from_documents_empty_documents(self, mock_generator_ollama):
        """Test answer generation with no documents."""
        result = mock_generator_ollama.generate_from_documents(
            "What is the Sample Program?",
            []
        )
        
        assert result['query'] == "What is the Sample Program?"
        assert "don't have enough context" in result['answer']
        assert result['sources'] == []
        assert result['num_sources'] == 0
    
    def test_generate_from_documents_llm_error(self, mock_generator_ollama, sample_documents):
        """Test answer generation with LLM error."""
        # Mock LLM error
        mock_generator_ollama.llm.invoke.side_effect = Exception("LLM service unavailable")
        
        result = mock_generator_ollama.generate_from_documents(
            "What is the Sample Program?",
            sample_documents
        )
        
        assert result['query'] == "What is the Sample Program?"
        assert 'error' in result
        assert 'LLM service unavailable' in result['error']
        assert result['answer'] is None
        assert result['sources'] == []
    
    def test_generate_from_documents_very_long_context(self, mock_generator_ollama):
        """Test answer generation with very long context."""
        long_documents = [
            Document(
                page_content="Very long content. " * 1000,  # Very long content
                metadata={'citation_source': 'Long Document', 'file_type': 'pdf'}
            )
        ]
        
        mock_response = Mock()
        mock_response.content = "Summary of long content."
        mock_generator_ollama.llm.invoke.return_value = mock_response
        
        result = mock_generator_ollama.generate_from_documents(
            "Summarize this content",
            long_documents
        )
        
        assert result['answer'] == "Summary of long content."
        # Content should be truncated in preview
        assert len(result['sources'][0]['content_preview']) <= 203  # 200 + "..."
    
    def test_context_preparation(self, mock_generator_ollama, sample_documents):
        """Test context preparation from documents."""
        context = mock_generator_ollama._prepare_context(sample_documents)
        
        assert "Source 1 (Program Overview):" in context
        assert "Source 2 (Program Curriculum):" in context
        assert "Source 3 (Program Benefits):" in context
        assert "Sample Educational Program" in context
        assert "machine learning" in context
        assert "mentorship" in context
    
    def test_answer_extraction_and_cleaning(self, mock_generator_ollama):
        """Test answer extraction and cleaning."""
        # Test with answer prefix
        response_with_prefix = "Context: ...\nQuestion: ...\nAnswer: The Sample Educational Program is great."
        cleaned = mock_generator_ollama._extract_answer(response_with_prefix, "test question")
        assert cleaned == "The Sample Educational Program is great."
        
        # Test without prefix
        response_without_prefix = "The Sample Educational Program is great."
        cleaned = mock_generator_ollama._extract_answer(response_without_prefix, "test question")
        assert cleaned == "The Sample Educational Program is great."
        
        # Test very short response
        short_response = "Yes."
        cleaned = mock_generator_ollama._extract_answer(short_response, "test question")
        assert "Yes." in cleaned
        assert "Note:" in cleaned  # Should add note for short responses
    
    def test_source_info_extraction(self, mock_generator_ollama, sample_documents):
        """Test source information extraction."""
        sources = mock_generator_ollama._extract_source_info(sample_documents)
        
        assert len(sources) == 3
        
        for i, source in enumerate(sources):
            assert source['citation'] == sample_documents[i].metadata['citation_source']
            assert source['file_type'] == sample_documents[i].metadata['file_type']
            assert 'content_preview' in source
            # Content should be truncated if longer than 200 chars
            if len(sample_documents[i].page_content) > 200:
                assert source['content_preview'].endswith("...")


class TestStreamingGeneration:
    """Test streaming answer generation."""
    
    @pytest.fixture
    def mock_streaming_generator(self, mock_ollama_config):
        """Create generator with mocked streaming capabilities."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama3'}]}
            mock_get.return_value = mock_response
            
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            
            generator = AnswerGenerator()
            return generator
    
    @pytest.mark.asyncio
    async def test_streaming_generation_ollama_success(self, mock_streaming_generator, sample_documents):
        """Test successful streaming generation with Ollama."""
        # Mock streaming response
        async def mock_astream(messages):
            tokens = ["The ", "Sample ", "Educational ", "Program ", "is ", "a ", "comprehensive ", "training ", "program."]
            for token in tokens:
                chunk = Mock()
                chunk.content = token
                yield chunk
        
        mock_streaming_generator.llm.astream = mock_astream
        
        # Collect streamed tokens
        tokens = []
        async for token in mock_streaming_generator.generate_streaming_from_documents(
            "What is the Sample Program?",
            sample_documents
        ):
            tokens.append(token)
        
        # Should receive all tokens
        full_response = "".join(tokens)
        assert "Sample Educational Program" in full_response
        assert "comprehensive" in full_response
        assert "training" in full_response
    
    @pytest.mark.asyncio
    async def test_streaming_generation_huggingface(self, mock_huggingface_config, sample_documents):
        """Test streaming generation with HuggingFace (simulated)."""
        with patch('src.rag.generator.load_config', return_value=mock_huggingface_config), \
             patch('transformers.pipeline') as mock_pipeline, \
             patch('src.rag.generator.HuggingFacePipeline') as mock_hf_pipeline:
            
            mock_pipeline.return_value = Mock()
            mock_llm = Mock()
            mock_hf_pipeline.return_value = mock_llm
            
            generator = AnswerGenerator()
            
            # Mock non-streaming response (will be streamed word by word)
            mock_response = "The Sample Educational Program is a comprehensive training program."
            
            with patch('asyncio.to_thread', return_value=mock_response):
                tokens = []
                async for token in generator.generate_streaming_from_documents(
                    "What is the Sample Program?",
                    sample_documents
                ):
                    tokens.append(token)
                
                full_response = "".join(tokens)
                assert full_response == mock_response
    
    @pytest.mark.asyncio
    async def test_streaming_generation_empty_documents(self, mock_streaming_generator):
        """Test streaming generation with no documents."""
        tokens = []
        async for token in mock_streaming_generator.generate_streaming_from_documents(
            "What is the Sample Program?",
            []
        ):
            tokens.append(token)
        
        full_response = "".join(tokens)
        assert "don't have enough context" in full_response
    
    @pytest.mark.asyncio
    async def test_streaming_generation_error(self, mock_streaming_generator, sample_documents):
        """Test streaming generation with error."""
        # Mock streaming error
        async def mock_astream_error(messages):
            raise Exception("Streaming failed")
        
        mock_streaming_generator.llm.astream = mock_astream_error
        
        tokens = []
        async for token in mock_streaming_generator.generate_streaming_from_documents(
            "What is the Sample Program?",
            sample_documents
        ):
            tokens.append(token)
        
        full_response = "".join(tokens)
        assert "Error generating response" in full_response
    
    @pytest.mark.asyncio
    async def test_streaming_generation_with_temperature_override(self, mock_streaming_generator, sample_documents):
        """Test streaming generation with temperature override."""
        async def mock_astream(messages):
            yield Mock(content="Test response with custom temperature.")
        
        mock_streaming_generator.llm.astream = mock_astream
        
        tokens = []
        async for token in mock_streaming_generator.generate_streaming_from_documents(
            "What is the Sample Program?",
            sample_documents,
            temperature=0.9
        ):
            tokens.append(token)
        
        full_response = "".join(tokens)
        assert "Test response" in full_response


class TestPromptTemplateHandling:
    """Test prompt template functionality."""
    
    def test_prompt_template_update(self, mock_generator_ollama):
        """Test updating prompt template."""
        new_template = "Custom template with {context} and {question}"
        new_variables = ['context', 'question']
        
        mock_generator_ollama.update_prompt_template(new_template, new_variables)
        
        assert mock_generator_ollama.prompt_template.template == new_template
        assert mock_generator_ollama.prompt_template.input_variables == new_variables
    
    def test_prompt_template_formatting(self, mock_generator_ollama, sample_documents):
        """Test prompt template formatting with context."""
        context = mock_generator_ollama._prepare_context(sample_documents)
        question = "What is the Sample Program?"
        
        formatted_prompt = mock_generator_ollama.prompt_template.format(
            context=context,
            question=question
        )
        
        assert question in formatted_prompt
        assert "Program Overview" in formatted_prompt
        assert "Sample Educational Program" in formatted_prompt


class TestRAGChainIntegration:
    """Test RAG chain integration functionality."""
    
    def test_create_rag_chain(self, mock_generator_ollama):
        """Test RAG chain creation."""
        mock_retriever = Mock()
        
        with patch('src.rag.generator.RetrievalQA') as mock_retrieval_qa:
            mock_retrieval_qa.from_chain_type.return_value = Mock()
            
            rag_chain = mock_generator_ollama.create_rag_chain(mock_retriever)
            
            assert rag_chain is not None
            mock_retrieval_qa.from_chain_type.assert_called_once_with(
                llm=mock_generator_ollama.llm,
                chain_type="stuff",
                retriever=mock_retriever,
                chain_type_kwargs={"prompt": mock_generator_ollama.prompt_template},
                return_source_documents=True
            )


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_ollama_server_connection_error(self, mock_ollama_config):
        """Test handling of Ollama server connection errors."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get', side_effect=requests.exceptions.ConnectionError("Connection refused")):
            
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama server"):
                AnswerGenerator()
    
    def test_ollama_server_timeout(self, mock_ollama_config):
        """Test handling of Ollama server timeout."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get', side_effect=requests.exceptions.Timeout("Request timeout")):
            
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama server"):
                AnswerGenerator()
    
    def test_ollama_server_error_response(self, mock_ollama_config):
        """Test handling of Ollama server error responses."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get:
            
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            with pytest.raises(RuntimeError, match="Ollama server not responding"):
                AnswerGenerator()
    
    def test_huggingface_model_loading_error(self, mock_huggingface_config):
        """Test handling of HuggingFace model loading errors."""
        with patch('src.rag.generator.load_config', return_value=mock_huggingface_config), \
             patch('transformers.pipeline', side_effect=[Exception("Model loading failed"), Exception("GPT-2 also failed")]):
            
            with pytest.raises(Exception, match="GPT-2 also failed"):
                AnswerGenerator()
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        invalid_config = {
            'llm': {},  # Missing required fields
            'prompts': {}
        }
        
        with patch('src.rag.generator.load_config', return_value=invalid_config):
            # Should use defaults
            generator = AnswerGenerator()
            assert generator.provider == 'huggingface'  # Default fallback
    
    def test_document_generation_with_malformed_metadata(self, mock_generator_ollama):
        """Test generation with documents having malformed metadata."""
        malformed_documents = [
            Document(
                page_content="Test content",
                metadata=None  # None metadata
            ),
            Document(
                page_content="More content",
                metadata={'citation_source': None, 'file_type': None}  # None values
            )
        ]
        
        mock_response = Mock()
        mock_response.content = "Generated answer despite malformed metadata."
        mock_generator_ollama.llm.invoke.return_value = mock_response
        
        result = mock_generator_ollama.generate_from_documents(
            "Test question",
            malformed_documents
        )
        
        # Should handle gracefully
        assert result['answer'] == "Generated answer despite malformed metadata."
        assert len(result['sources']) == 2
        # Should provide fallback values for missing metadata
        for source in result['sources']:
            assert 'citation' in source
            assert 'file_type' in source
            assert 'content_preview' in source


class TestPerformance:
    """Test performance characteristics."""
    
    def test_context_preparation_performance(self, mock_generator_ollama):
        """Test context preparation performance with many documents."""
        # Create many documents
        many_documents = [
            Document(
                page_content=f"Document {i} content with some information about topic {i}.",
                metadata={'citation_source': f'Source {i}', 'file_type': 'pdf'}
            ) for i in range(100)
        ]
        
        import time
        start_time = time.time()
        context = mock_generator_ollama._prepare_context(many_documents)
        end_time = time.time()
        
        assert end_time - start_time < 1.0  # Should complete within 1 second
        assert "Source 1" in context
        assert "Source 100" in context
        assert len(context) > 0
    
    def test_source_extraction_performance(self, mock_generator_ollama):
        """Test source extraction performance with many documents."""
        many_documents = [
            Document(
                page_content=f"Document {i} content " * 50,  # Longer content
                metadata={'citation_source': f'Source {i}', 'file_type': 'html'}
            ) for i in range(50)
        ]
        
        import time
        start_time = time.time()
        sources = mock_generator_ollama._extract_source_info(many_documents)
        end_time = time.time()
        
        assert end_time - start_time < 1.0  # Should complete within 1 second
        assert len(sources) == 50
        # All sources should have truncated content
        for source in sources:
            assert len(source['content_preview']) <= 203  # 200 + "..."
    
    @pytest.mark.asyncio
    async def test_streaming_performance(self, mock_streaming_generator, sample_documents):
        """Test streaming generation performance."""
        # Mock fast streaming response
        async def mock_fast_astream(messages):
            for i in range(100):
                chunk = Mock()
                chunk.content = f"word{i} "
                yield chunk
        
        mock_streaming_generator.llm.astream = mock_fast_astream
        
        import time
        start_time = time.time()
        
        token_count = 0
        async for token in mock_streaming_generator.generate_streaming_from_documents(
            "Generate a long response",
            sample_documents
        ):
            token_count += 1
        
        end_time = time.time()
        
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        assert token_count > 0


class TestConfigurationHandling:
    """Test configuration handling edge cases."""
    
    def test_missing_prompt_config(self, mock_ollama_config):
        """Test handling of missing prompt configuration."""
        # Remove prompts section
        del mock_ollama_config['prompts']
        
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama3'}]}
            mock_get.return_value = mock_response
            mock_ollama.return_value = Mock()
            
            generator = AnswerGenerator()
            
            # Should use default template
            assert generator.prompt_template is not None
            assert 'context' in generator.prompt_template.input_variables
            assert 'question' in generator.prompt_template.input_variables
    
    def test_environment_variable_override(self, mock_ollama_config):
        """Test environment variable override for base URL."""
        with patch('src.rag.generator.load_config', return_value=mock_ollama_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama, \
             patch.dict('os.environ', {'OLLAMA_BASE_URL': 'http://custom-ollama:11434'}):
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama3'}]}
            mock_get.return_value = mock_response
            mock_ollama.return_value = Mock()
            
            generator = AnswerGenerator()
            
            assert generator.base_url == 'http://custom-ollama:11434'
    
    def test_partial_llm_config(self):
        """Test handling of partial LLM configuration."""
        partial_config = {
            'llm': {
                'provider': 'ollama',
                'model': 'llama3'
                # Missing temperature, max_tokens, base_url
            },
            'prompts': {
                'rag_template': 'Test template: {context} {question}',
                'input_variables': ['context', 'question']
            }
        }
        
        with patch('src.rag.generator.load_config', return_value=partial_config), \
             patch('src.rag.generator.requests.get') as mock_get, \
             patch('src.rag.generator.ChatOllama') as mock_ollama:
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama3'}]}
            mock_get.return_value = mock_response
            mock_ollama.return_value = Mock()
            
            generator = AnswerGenerator()
            
            # Should use defaults for missing values
            assert generator.temperature == 0.7  # Default
            assert generator.max_length == 512  # Default
            assert generator.base_url == 'http://localhost:11434'  # Default
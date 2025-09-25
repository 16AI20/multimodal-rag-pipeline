"""
Answer generation component for RAG pipeline.
Handles LLM initialization and response generation.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain.schema import Document
import asyncio
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ..utils import load_config
from ..interfaces.base_interfaces import BaseGenerator, GenerationError
from ..utils.validation import validate_runtime_parameters, ParameterValidator

logger = logging.getLogger(__name__)


class AnswerGenerator(BaseGenerator):
    """
    Handles answer generation using language models.
    
    This class implements the BaseGenerator interface and provides methods for
    generating answers from retrieved documents using various LLM providers
    including Ollama and HuggingFace models.
    
    Attributes:
        config: System configuration loaded via Hydra
        provider: LLM provider name (ollama, huggingface)
        llm_model: Name of the language model being used
        temperature: Sampling temperature for generation
        max_length: Maximum length of generated responses
        llm: Initialized language model instance
        prompt_template: Configured prompt template for RAG
    
    Example:
        >>> generator = AnswerGenerator(config_path="conf")
        >>> documents = [Document(page_content="ML is...", metadata={})]
        >>> result = generator.generate_answer("What is ML?", documents)
        >>> print(result["answer"])
    """
    
    def __init__(self, 
                 config_path: str = "conf",
                 llm_model: str = None,
                 temperature: float = None,
                 max_length: int = None):
        """
        Initialize the answer generator with configuration and model setup.
        
        Args:
            config_path: Path to configuration directory containing Hydra configs
            llm_model: Override for LLM model name (optional)
            temperature: Override for sampling temperature (0.0-2.0, optional)
            max_length: Override for maximum response length (optional)
            
        Raises:
            ConfigurationError: When configuration is invalid or missing
            RuntimeError: When LLM initialization fails
            
        Example:
            >>> generator = AnswerGenerator(
            ...     config_path="conf",
            ...     llm_model="llama3.1:8b", 
            ...     temperature=0.3
            ... )
        """
        # Load configuration using Hydra
        self.config = load_config(config_path)
        
        # Debug: Log loaded config
        logger.debug(f"Loaded config: {self.config}")
        
        # Use config values or provided overrides (Hydra configs use dot notation)
        llm_config = self.config.llm
        logger.debug(f"LLM config section: {llm_config}")
        
        self.provider = llm_config.get('provider', 'huggingface')
        self.llm_model = llm_model or llm_config.get('model', 'microsoft/DialoGPT-medium')
        self.temperature = temperature or llm_config.get('temperature', 0.7)
        self.max_length = max_length or llm_config.get('max_tokens', 512)
        
        # Use environment variable override for base_url (for container compatibility)
        import os
        self.base_url = os.getenv('OLLAMA_BASE_URL', llm_config.get('base_url', 'http://localhost:11434'))
        
        logger.info(f"Config - Provider: {self.provider}, Model: {self.llm_model}")
        
        # Initialize LLM
        logger.info(f"Loading {self.provider} model: {self.llm_model}")
        self.llm = self._initialize_llm()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        logger.info("Answer generator initialized successfully")
    
    def _initialize_llm(self):
        """Initialize the language model based on provider."""
        if self.provider == "ollama":
            # No fallback for Ollama - fail fast if there are issues
            return self._initialize_ollama_llm()
        else:
            return self._initialize_huggingface_llm()
    
    def _initialize_ollama_llm(self) -> ChatOllama:
        """Initialize Ollama LLM."""
        try:
            # First check if Ollama server is running
            import requests
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise Exception(f"Ollama server not responding (status: {response.status_code})")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Cannot connect to Ollama server at {self.base_url}. Please start Ollama with 'ollama serve'")
            
            # Check if the model exists
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            if self.llm_model not in model_names:
                available_models = ', '.join(model_names) if model_names else 'None'
                raise Exception(f"Model '{self.llm_model}' not found in Ollama. Available models: {available_models}")
            
            # Initialize the LLM
            llm = ChatOllama(
                model=self.llm_model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_length
            )
            
            logger.info(f"Ollama LLM initialized successfully: {self.llm_model}")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
            logger.error("💡 Make sure to:")
            logger.error("   1. Start Ollama: ollama serve")
            logger.error(f"   2. Have model '{self.llm_model}' available: ollama pull {self.llm_model}")
            raise RuntimeError(f"Ollama LLM initialization failed: {e}")
    
    def _initialize_huggingface_llm(self) -> HuggingFacePipeline:
        """Initialize HuggingFace LLM."""
        try:
            from transformers import pipeline
            
            # Try the specified model first
            hf_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                device_map="cpu",
                max_length=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9
            )
            
            return HuggingFacePipeline(pipeline=hf_pipeline)
            
        except Exception as e:
            logger.warning(f"Error loading {self.llm_model}: {e}")
            logger.info("Falling back to GPT-2...")
            
            # Fallback to GPT-2
            hf_pipeline = pipeline(
                "text-generation",
                model="gpt2",
                device_map="cpu",
                max_length=min(self.max_length, 256),  # GPT-2 has smaller context
                do_sample=True,
                temperature=self.temperature
            )
            
            return HuggingFacePipeline(pipeline=hf_pipeline)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for RAG from config."""
        # Get prompt configuration (Hydra dot notation)
        prompt_config = self.config.prompts
        
        # Use configured template or fallback to default
        template = prompt_config.get('rag_template', """You are an AI assistant helping answer questions based on the provided document context.

Use the following context information to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, informative answer based on the context above. Include relevant details.""")
        
        # Get input variables from config or use defaults
        input_variables = prompt_config.get('input_variables', ["context", "question"])
        
        logger.info(f"Using prompt template from config: {len(template)} characters")
        
        return PromptTemplate(
            template=template,
            input_variables=input_variables
        )
    
    def generate_from_documents(self, 
                              query: str, 
                              documents: List[Document]) -> Dict[str, Any]:
        """
        Generate an answer from a list of documents.
        
        Args:
            query: The user's question
            documents: List of relevant documents for context
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            if not documents:
                return {
                    'query': query,
                    'answer': "I don't have enough context to answer this question.",
                    'sources': [],
                    'num_sources': 0
                }
            
            # Prepare context from documents
            context = self._prepare_context(documents)
            
            # Generate answer using the prompt template
            prompt = self.prompt_template.format(context=context, question=query)
            
            # Generate response
            logger.debug("Generating answer...")
            if self.provider == "ollama":
                # For Ollama, use invoke method
                response = self.llm.invoke(prompt)
                # Handle response object
                if hasattr(response, 'content'):
                    raw_answer = response.content
                else:
                    raw_answer = str(response)
            else:
                # For HuggingFace, use the old method
                response = self.llm(prompt)
                raw_answer = response
            
            # Extract and clean the answer
            answer = self._extract_answer(raw_answer, query)
            
            # Prepare source information
            sources = self._extract_source_info(documents)
            
            return {
                'query': query,
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources)
            }
            
        except Exception as e:
            error_context = {
                "query": query[:200] if query else "None",
                "num_documents": len(documents) if documents else 0,
                "provider": getattr(self, 'provider', 'unknown'),
                "llm_model": getattr(self, 'llm_model', 'unknown'),
                "temperature": getattr(self, 'temperature', 'unknown'),
                "max_length": getattr(self, 'max_length', 'unknown')
            }
            logger.error(f"Answer generation failed: {str(e)}", extra={"context": error_context})
            raise GenerationError(f"Failed to generate answer: {str(e)}") from e
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare formatted context string from retrieved documents.
        
        Combines multiple documents into a well-structured context string
        for LLM input, with source attribution and content truncation.
        
        Args:
            documents: List of LangChain Document objects with content and metadata.
        
        Returns:
            str: Formatted context string with numbered sources and content.
        
        Example:
            Returns format like:
            "Source 1 (document.pdf):\\nContent here...\\n\\nSource 2 (webpage.html):\\nMore content..."
        """
        logger.debug("Preparing context from %d documents", len(documents))
        
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('citation_source', f'Document {i}')
            # Limit content length to prevent prompt overflow
            content = doc.page_content[:500]
            if len(doc.page_content) > 500:
                content += "..."
            
            context_part = f"Source {i} ({source}):\\n{content}"
            context_parts.append(context_part)
            total_chars += len(context_part)
        
        context = "\\n\\n".join(context_parts)
        logger.debug("Prepared context: %d characters from %d sources", total_chars, len(documents))
        
        return context
    
    def _extract_answer(self, response: str, query: str) -> str:
        """Extract and clean the generated answer."""
        # Remove the prompt from the response if it's included
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response.strip()
        
        # Clean up the response
        answer = answer.replace("\\n\\n", "\\n").strip()
        
        # If the answer is too short or seems incomplete, add a note
        if len(answer) < 20:
            answer = f"{answer}\\n\\n(Note: Generated response was brief. You may want to rephrase your question or check if relevant documents are available.)"
        
        return answer
    
    def _extract_source_info(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        
        for doc in documents:
            source_info = {
                'citation': doc.metadata.get('citation_source', 'Unknown'),
                'file_type': doc.metadata.get('file_type', 'unknown'),
                'content_preview': (
                    doc.page_content[:200] + "..." 
                    if len(doc.page_content) > 200 
                    else doc.page_content
                )
            }
            sources.append(source_info)
        
        return sources
    
    def create_rag_chain(self, retriever) -> RetrievalQA:
        """
        Create a complete RAG chain with the given retriever.
        
        Args:
            retriever: Document retriever instance
            
        Returns:
            Configured RetrievalQA chain
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def update_prompt_template(self, template: str, input_variables: List[str]) -> None:
        """
        Update the prompt template.
        
        Args:
            template: New prompt template string
            input_variables: List of input variables in the template
        """
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        logger.info("Prompt template updated")
    
    
    async def generate_streaming_from_documents(self, 
                                              query: str, 
                                              documents: List[Document], 
                                              temperature: float = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer from the provided documents.
        This uses the same quality logic as generate_from_documents but streams the final answer.
        
        Args:
            query: User's question
            documents: List of relevant documents
            temperature: Override for sampling temperature
            
        Yields:
            Tokens as they are generated
        """
        try:
            if not documents:
                yield "I don't have enough context to answer this question."
                return
            
            # Use the same context preparation as non-streaming
            context = self._prepare_context(documents)
            prompt = self.prompt_template.format(context=context, question=query)
            
            # Use appropriate streaming method based on provider
            if self.provider == "ollama":
                # Stream from Ollama and apply the same cleaning as non-streaming
                full_response = ""
                async for token in self._stream_ollama_raw(prompt, temperature):
                    full_response += token
                
                # Apply the same answer extraction and cleaning as non-streaming
                cleaned_answer = self._extract_answer(full_response, query)
                
                # Stream the cleaned answer word by word
                words = cleaned_answer.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield f" {word}"
                    await asyncio.sleep(0.03)  # Small delay for streaming effect
            else:
                # For HuggingFace, generate full response then stream
                response = await asyncio.to_thread(self.llm, prompt)
                cleaned_answer = self._extract_answer(response, query)
                
                # Stream the cleaned answer
                words = cleaned_answer.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield f" {word}"
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error generating response: {str(e)}"
    
    async def _stream_ollama_raw(self, prompt: str, temperature: float = None) -> AsyncGenerator[str, None]:
        """Stream raw tokens from Ollama LLM without any filtering."""
        try:
            # Use the temperature override if provided
            temp = temperature if temperature is not None else self.temperature
            
            # For ChatOllama, we need to use astream
            from langchain.schema import HumanMessage
            
            message = HumanMessage(content=prompt)
            
            # Stream the response
            async for chunk in self.llm.astream([message]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            yield f"[Ollama streaming error: {str(e)}]"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check generator health status and model connectivity.
        
        Returns:
            Dictionary containing health status and diagnostic information:
            - status: Overall health status (healthy/unhealthy/degraded)
            - llm_provider: Language model provider status
            - model_info: Information about loaded model
            - diagnostics: Detailed diagnostic information
            
        Example:
            >>> generator = AnswerGenerator()
            >>> health = generator.health_check()
            >>> print(f"Status: {health['status']}")
        """
        diagnostics = {
            "timestamp": time.time(),
            "component": "AnswerGenerator", 
            "version": "1.0.0",
            "provider": getattr(self, 'provider', 'unknown'),
            "model": getattr(self, 'llm_model', 'unknown')
        }
        
        try:
            logger.debug("Performing generator health check")
            
            # Test LLM connection with simple query
            with performance_context("llm_health_check"):
                if self.provider == "ollama":
                    # Test Ollama connection
                    test_response = self.llm.invoke("Test")
                    if hasattr(test_response, 'content'):
                        diagnostics["test_response_length"] = len(test_response.content)
                    else:
                        diagnostics["test_response_length"] = len(str(test_response))
                else:
                    # Test HuggingFace model
                    test_response = self.llm("Test")
                    diagnostics["test_response_length"] = len(str(test_response))
                
                diagnostics["llm_responsive"] = True
            
            health_result = {
                "status": "healthy",
                "llm_provider": f"{self.provider}_connected",
                "model_info": {
                    "provider": self.provider,
                    "model": self.llm_model,
                    "temperature": self.temperature,
                    "max_length": self.max_length
                },
                "diagnostics": diagnostics
            }
            
            logger.info("Generator health check completed - Status: healthy")
            return health_result
            
        except Exception as e:
            error_context = {
                "component": "AnswerGenerator",
                "error_type": type(e).__name__,
                "provider": getattr(self, 'provider', 'unknown'),
                "model": getattr(self, 'llm_model', 'unknown'),
                "base_url": getattr(self, 'base_url', 'unknown')
            }
            logger.error(f"Generator health check failed: {str(e)}", extra={"context": error_context})
            
            return {
                "status": "unhealthy",
                "llm_provider": f"{self.provider}_error",
                "error": str(e),
                "diagnostics": {**diagnostics, "error_context": error_context}
            }
"""
Configuration validation utilities using Pydantic.

This module provides comprehensive validation for system configuration,
ensuring that all required parameters are present and correctly typed
before system initialization.

Classes:
    EmbeddingsConfig: Validation for embedding model configuration
    LLMConfig: Validation for language model configuration  
    VectorDBConfig: Validation for vector database configuration
    SystemConfig: Top-level system configuration validation
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator, root_validator
from pathlib import Path
import os


class EmbeddingsConfig(BaseModel):
    """Configuration validation for embedding models."""
    
    model: str = Field(..., description="HuggingFace embedding model name")
    normalize_embeddings: bool = Field(True, description="Whether to L2 normalize embeddings")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional encoding parameters")
    reranking_enabled: bool = Field(False, description="Enable cross-encoder reranking")
    reranking_model: Optional[str] = Field(None, description="Cross-encoder model for reranking")
    
    @validator('model')
    def validate_model_format(cls, v):
        """Validate embedding model name format."""
        if not isinstance(v, str) or len(v) == 0:
            raise ValueError("Embedding model name must be a non-empty string")
        if '/' not in v:
            raise ValueError("Embedding model should be in format 'organization/model-name'")
        return v
    
    @validator('reranking_model')
    def validate_reranking_model(cls, v, values):
        """Validate reranking model when reranking is enabled."""
        if values.get('reranking_enabled', False) and not v:
            raise ValueError("reranking_model is required when reranking_enabled is True")
        return v


class OllamaConfig(BaseModel):
    """Configuration validation for Ollama LLM."""
    
    base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    model: str = Field(..., description="Ollama model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum response length")
    timeout: float = Field(30.0, gt=0, description="Request timeout in seconds")
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Validate Ollama base URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip('/')  # Remove trailing slash


class HuggingFaceConfig(BaseModel):
    """Configuration validation for HuggingFace LLM."""
    
    model: str = Field(..., description="HuggingFace model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_length: int = Field(512, ge=1, le=4096, description="Maximum sequence length")
    device_map: str = Field("auto", description="Device mapping strategy")
    trust_remote_code: bool = Field(False, description="Trust remote code in model")


class LLMConfig(BaseModel):
    """Configuration validation for language models."""
    
    provider: str = Field(..., description="LLM provider (ollama, huggingface)")
    ollama: Optional[OllamaConfig] = Field(None, description="Ollama configuration")
    huggingface: Optional[HuggingFaceConfig] = Field(None, description="HuggingFace configuration")
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate LLM provider is supported."""
        supported_providers = ['ollama', 'huggingface']
        if v not in supported_providers:
            raise ValueError(f"provider must be one of {supported_providers}")
        return v
    
    @root_validator
    def validate_provider_config(cls, values):
        """Validate that provider-specific configuration is present."""
        provider = values.get('provider')
        if provider == 'ollama' and not values.get('ollama'):
            raise ValueError("ollama configuration is required when provider is 'ollama'")
        elif provider == 'huggingface' and not values.get('huggingface'):
            raise ValueError("huggingface configuration is required when provider is 'huggingface'")
        return values


class VectorDBConfig(BaseModel):
    """Configuration validation for vector database."""
    
    path: str = Field(..., description="Path to vector database storage")
    collection_name: str = Field(..., description="Collection name in vector database")
    similarity_function: str = Field("cosine", description="Similarity function for search")
    persist_directory: bool = Field(True, description="Whether to persist database to disk")
    
    @validator('path')
    def validate_path_format(cls, v):
        """Validate database path format."""
        if not isinstance(v, str) or len(v) == 0:
            raise ValueError("Database path must be a non-empty string")
        return v
    
    @validator('collection_name')
    def validate_collection_name(cls, v):
        """Validate collection name format."""
        if not isinstance(v, str) or len(v) == 0:
            raise ValueError("Collection name must be a non-empty string")
        # ChromaDB naming constraints
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Collection name must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @validator('similarity_function')
    def validate_similarity_function(cls, v):
        """Validate similarity function is supported."""
        supported_functions = ['cosine', 'euclidean', 'manhattan']
        if v not in supported_functions:
            raise ValueError(f"similarity_function must be one of {supported_functions}")
        return v


class DocumentProcessingConfig(BaseModel):
    """Configuration validation for document processing."""
    
    resource_dirs: Dict[str, str] = Field(..., description="Resource directories for each document type")
    chunk_sizes: Dict[str, int] = Field(..., description="Chunk sizes for different document types")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Chunk overlap in characters")
    
    @validator('resource_dirs')
    def validate_resource_dirs(cls, v):
        """Validate resource directories exist or can be created."""
        required_types = ['html', 'pdf', 'docx', 'csv', 'audio', 'images']
        for doc_type in required_types:
            if doc_type not in v:
                raise ValueError(f"Missing resource directory for document type: {doc_type}")
        return v
    
    @validator('chunk_sizes')
    def validate_chunk_sizes(cls, v):
        """Validate chunk sizes are reasonable."""
        for doc_type, size in v.items():
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Chunk size for {doc_type} must be a positive integer")
            if size < 100:
                raise ValueError(f"Chunk size for {doc_type} is too small (minimum 100)")
            if size > 5000:
                raise ValueError(f"Chunk size for {doc_type} is too large (maximum 5000)")
        return v


class SystemConfig(BaseModel):
    """Top-level system configuration validation."""
    
    device: str = Field("auto", description="Compute device (cpu, cuda, mps, auto)")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    embeddings: EmbeddingsConfig = Field(..., description="Embedding model configuration")
    llm: LLMConfig = Field(..., description="Language model configuration")
    vectordb: VectorDBConfig = Field(..., description="Vector database configuration")
    document_processing: DocumentProcessingConfig = Field(..., description="Document processing configuration")
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device option is supported."""
        supported_devices = ['cpu', 'cuda', 'mps', 'auto']
        if v not in supported_devices:
            raise ValueError(f"device must be one of {supported_devices}")
        return v


def validate_config(config_dict: Dict[str, Any]) -> SystemConfig:
    """
    Validate system configuration using Pydantic models.
    
    Args:
        config_dict: Raw configuration dictionary from Hydra/YAML
        
    Returns:
        Validated SystemConfig object
        
    Raises:
        ConfigurationError: When validation fails with detailed error messages
    """
    try:
        return SystemConfig(**config_dict)
    except Exception as e:
        from .base_interfaces import ConfigurationError
        raise ConfigurationError(f"Configuration validation failed: {str(e)}") from e


def validate_runtime_parameters(**kwargs) -> Dict[str, Any]:
    """
    Validate runtime parameters for RAG operations.
    
    Args:
        **kwargs: Runtime parameters to validate
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValueError: When parameter validation fails
    """
    validated = {}
    
    # Validate k parameter
    if 'k' in kwargs:
        k = kwargs['k']
        if not isinstance(k, int) or k <= 0 or k > 50:
            raise ValueError("k must be an integer between 1 and 50")
        validated['k'] = k
    
    # Validate temperature parameter
    if 'temperature' in kwargs:
        temp = kwargs['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ValueError("temperature must be a number between 0 and 2")
        validated['temperature'] = float(temp)
    
    # Validate max_tokens parameter
    if 'max_tokens' in kwargs:
        max_tokens = kwargs['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > 4096:
            raise ValueError("max_tokens must be an integer between 1 and 4096")
        validated['max_tokens'] = max_tokens
    
    # Validate file_type parameter
    if 'file_type' in kwargs:
        file_type = kwargs['file_type']
        if file_type is not None:
            supported_types = ['html', 'pdf', 'docx', 'csv', 'audio', 'images']
            if file_type not in supported_types:
                raise ValueError(f"file_type must be one of {supported_types} or None")
        validated['file_type'] = file_type
    
    return validated


class ParameterValidator:
    """Decorator class for method parameter validation."""
    
    @staticmethod
    def validate_query_parameters(func):
        """Decorator to validate query-related parameters."""
        def wrapper(*args, **kwargs):
            # Validate common query parameters
            validated_kwargs = validate_runtime_parameters(**kwargs)
            kwargs.update(validated_kwargs)
            return func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def validate_required_params(*required_params):
        """Decorator to validate required parameters are present."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                missing_params = []
                for param in required_params:
                    if param not in kwargs or kwargs[param] is None:
                        missing_params.append(param)
                
                if missing_params:
                    raise ValueError(f"Missing required parameters: {missing_params}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
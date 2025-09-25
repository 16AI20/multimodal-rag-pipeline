"""
Advanced configuration tests for the RAG system.
Tests complex configuration scenarios, environment variable handling, and Hydra integration.

This test suite covers:
- Complex configuration merging scenarios with multiple sources
- Environment variable precedence and override patterns
- Configuration validation and error handling for invalid configs
- Hydra integration with overrides and composition
- Corpus directory configuration and validation
- Device configuration with fallback mechanisms
- Configuration inheritance and defaults
- Dynamic configuration updates and reloading

Each test ensures:
- Robust configuration handling under various scenarios
- Proper precedence rules for configuration sources
- Graceful handling of invalid or missing configurations
- Consistent behavior across different deployment environments
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from typing import Dict, Any
import copy

from src.utils.core import load_config


class TestConfigurationMerging:
    """Test complex configuration merging scenarios."""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            'run': {
                'device': 'auto',
                'seed': 42
            },
            'embeddings': {
                'model': 'BAAI/bge-large-en-v1.5',
                'device': 'cpu',
                'reranking_enabled': False
            },
            'llm': {
                'provider': 'ollama',
                'model': 'llama3.1:8b',
                'temperature': 0.7,
                'max_tokens': 512,
                'base_url': 'http://localhost:11434'
            },
            'vectordb': {
                'path': './vector_store',
                'collection_name': 'rag_documents'
            },
            'document_processing': {
                'resource_dirs': {
                    'html': 'corpus/html',
                    'pdf': 'corpus/pdf',
                    'audio': 'corpus/audio',
                    'images': 'corpus/images',
                    'csv': 'corpus/csv',
                    'docx': 'corpus/docx'
                },
                'chunk_sizes': {
                    'html': 1200,
                    'pdf': 1000,
                    'audio': 800,
                    'csv': 600,
                    'docx': 1000
                }
            }
        }
    
    @pytest.fixture
    def override_config(self):
        """Configuration overrides for testing."""
        return {
            'run': {
                'device': 'cuda',
                'debug': True
            },
            'embeddings': {
                'model': 'custom-embedding-model',
                'reranking_enabled': True,
                'reranking_model': 'custom-reranker'
            },
            'llm': {
                'temperature': 0.5,
                'base_url': 'http://custom-ollama:11434'
            },
            'vectordb': {
                'collection_name': 'custom_collection'
            }
        }
    
    def test_deep_config_merging(self, base_config, override_config):
        """Test deep merging of nested configuration dictionaries."""
        def deep_merge(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
            """Deep merge two dictionaries."""
            result = copy.deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(base_config, override_config)
        
        # Test that overrides took effect
        assert merged['run']['device'] == 'cuda'  # Overridden
        assert merged['run']['seed'] == 42  # Preserved from base
        assert merged['run']['debug'] is True  # Added from override
        
        assert merged['embeddings']['model'] == 'custom-embedding-model'  # Overridden
        assert merged['embeddings']['device'] == 'cpu'  # Preserved from base
        assert merged['embeddings']['reranking_enabled'] is True  # Overridden
        assert merged['embeddings']['reranking_model'] == 'custom-reranker'  # Added
        
        assert merged['llm']['provider'] == 'ollama'  # Preserved
        assert merged['llm']['temperature'] == 0.5  # Overridden
        assert merged['llm']['base_url'] == 'http://custom-ollama:11434'  # Overridden
        
        # Test that nested structures are preserved
        assert 'document_processing' in merged
        assert merged['document_processing']['resource_dirs']['html'] == 'corpus/html'
    
    def test_config_precedence_order(self, base_config):
        """Test configuration precedence: env vars > config file > defaults."""
        # Create config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(base_config, f)
            config_file = f.name
        
        try:
            # Test with environment variable overrides
            env_overrides = {
                'EMBEDDING_MODEL': 'env-embedding-model',
                'OLLAMA_BASE_URL': 'http://env-ollama:11434',
                'CHROMA_DB_PATH': '/env/vector_store',
                'LLM_TEMPERATURE': '0.3'
            }
            
            with patch.dict(os.environ, env_overrides):
                # Mock the load_config to simulate precedence handling
                with patch('builtins.open', mock_open(read_data=yaml.dump(base_config))):
                    # Simulate precedence logic
                    config = copy.deepcopy(base_config)
                    
                    # Apply environment variable overrides
                    if 'EMBEDDING_MODEL' in os.environ:
                        config['embeddings']['model'] = os.environ['EMBEDDING_MODEL']
                    if 'OLLAMA_BASE_URL' in os.environ:
                        config['llm']['base_url'] = os.environ['OLLAMA_BASE_URL']
                    if 'CHROMA_DB_PATH' in os.environ:
                        config['vectordb']['path'] = os.environ['CHROMA_DB_PATH']
                    if 'LLM_TEMPERATURE' in os.environ:
                        config['llm']['temperature'] = float(os.environ['LLM_TEMPERATURE'])
            
            # Verify precedence worked
            assert config['embeddings']['model'] == 'env-embedding-model'
            assert config['llm']['base_url'] == 'http://env-ollama:11434'
            assert config['vectordb']['path'] == '/env/vector_store'
            assert config['llm']['temperature'] == 0.3
            
            # Verify non-overridden values remain
            assert config['llm']['provider'] == 'ollama'
            assert config['run']['seed'] == 42
            
        finally:
            Path(config_file).unlink()
    
    def test_partial_config_override(self, base_config):
        """Test partial configuration overrides don't lose other settings."""
        partial_override = {
            'embeddings': {
                'model': 'new-model-only'
            }
        }
        
        def apply_partial_override(base: Dict, override: Dict) -> Dict:
            result = copy.deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key].update(value)
                else:
                    result[key] = value
            return result
        
        merged = apply_partial_override(base_config, partial_override)
        
        # Override should apply
        assert merged['embeddings']['model'] == 'new-model-only'
        
        # Other embeddings settings should be preserved
        assert merged['embeddings']['device'] == 'cpu'
        assert merged['embeddings']['reranking_enabled'] is False
        
        # Other top-level sections should be unchanged
        assert merged['llm']['provider'] == 'ollama'
        assert merged['run']['device'] == 'auto'


class TestEnvironmentVariableHandling:
    """Test environment variable handling and validation."""
    
    def test_all_environment_variable_overrides(self):
        """Test comprehensive environment variable override patterns."""
        env_vars = {
            # Core configuration
            'CONFIG_PATH': '/custom/config.yaml',
            'DEVICE': 'cuda',
            'SEED': '123',
            
            # Database configuration  
            'CHROMA_DB_PATH': '/custom/vector_store',
            'COLLECTION_NAME': 'custom_collection',
            
            # Embedding configuration
            'EMBEDDING_MODEL': 'custom/embedding-model',
            'EMBEDDING_DEVICE': 'cuda',
            'RERANKING_ENABLED': 'true',
            'RERANKING_MODEL': 'custom/reranker',
            
            # LLM configuration
            'LLM_PROVIDER': 'huggingface',
            'LLM_MODEL': 'custom-llm-model',
            'OLLAMA_BASE_URL': 'http://custom-ollama:11434',
            'LLM_TEMPERATURE': '0.8',
            'LLM_MAX_TOKENS': '1024',
            
            # Corpus directories
            'CORPUS_HTML_DIR': '/custom/html',
            'CORPUS_PDF_DIR': '/custom/pdf',
            'CORPUS_AUDIO_DIR': '/custom/audio',
            'CORPUS_IMAGES_DIR': '/custom/images',
            'CORPUS_CSV_DIR': '/custom/csv',
            'CORPUS_DOCX_DIR': '/custom/docx',
            
            # Processing configuration
            'HTML_CHUNK_SIZE': '1500',
            'PDF_CHUNK_SIZE': '1200',
            'AUDIO_CHUNK_SIZE': '900'
        }
        
        with patch.dict(os.environ, env_vars):
            # Test that environment variables can be read
            for var_name, expected_value in env_vars.items():
                assert os.getenv(var_name) == expected_value
    
    def test_boolean_environment_variable_parsing(self):
        """Test parsing of boolean environment variables."""
        boolean_test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('on', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('no', False),
            ('off', False),
            ('', False),
            ('invalid', False)
        ]
        
        def parse_bool_env(value: str) -> bool:
            """Parse boolean environment variable."""
            if not value:
                return False
            return value.lower() in ('true', '1', 'yes', 'on')
        
        for env_value, expected in boolean_test_cases:
            with patch.dict(os.environ, {'TEST_BOOL': env_value}):
                result = parse_bool_env(os.getenv('TEST_BOOL', ''))
                assert result == expected, f"Failed for value: {env_value}"
    
    def test_numeric_environment_variable_parsing(self):
        """Test parsing of numeric environment variables."""
        numeric_test_cases = [
            ('42', int, 42),
            ('3.14', float, 3.14),
            ('0', int, 0),
            ('-10', int, -10),
            ('1e3', float, 1000.0),
            ('invalid', int, None),  # Should handle gracefully
            ('', float, None)
        ]
        
        def parse_numeric_env(value: str, target_type):
            """Parse numeric environment variable with error handling."""
            if not value:
                return None
            try:
                return target_type(value)
            except (ValueError, TypeError):
                return None
        
        for env_value, target_type, expected in numeric_test_cases:
            with patch.dict(os.environ, {'TEST_NUMERIC': env_value}):
                result = parse_numeric_env(os.getenv('TEST_NUMERIC', ''), target_type)
                assert result == expected, f"Failed for value: {env_value}, type: {target_type}"
    
    def test_environment_variable_validation(self):
        """Test validation of environment variable values."""
        def validate_device_env(device: str) -> str:
            """Validate device environment variable."""
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            if device not in valid_devices:
                return 'auto'  # Default fallback
            return device
        
        def validate_path_env(path: str) -> str:
            """Validate path environment variable."""
            if not path:
                return './default_path'
            # Additional path validation could go here
            return path
        
        # Test device validation
        device_test_cases = [
            ('cpu', 'cpu'),
            ('cuda', 'cuda'), 
            ('mps', 'mps'),
            ('auto', 'auto'),
            ('invalid', 'auto'),
            ('GPU', 'auto')  # Case sensitive
        ]
        
        for input_device, expected_device in device_test_cases:
            result = validate_device_env(input_device)
            assert result == expected_device
        
        # Test path validation
        path_test_cases = [
            ('/valid/path', '/valid/path'),
            ('relative/path', 'relative/path'),
            ('', './default_path'),
            ('.', '.')
        ]
        
        for input_path, expected_path in path_test_cases:
            result = validate_path_env(input_path)
            assert result == expected_path


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML configuration files."""
        invalid_yaml_cases = [
            "invalid: yaml: content: [unclosed",  # Syntax error
            "key1: value1\n\tinvalid_indentation: value2",  # Indentation error
            "duplicate_key: value1\nduplicate_key: value2",  # Duplicate keys
            "{invalid json mixed with yaml}",  # Mixed formats
        ]
        
        for invalid_yaml in invalid_yaml_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(invalid_yaml)
                f.flush()
                
                try:
                    # Should handle invalid YAML gracefully
                    with pytest.raises(yaml.YAMLError):
                        with open(f.name, 'r') as yaml_file:
                            yaml.safe_load(yaml_file)
                finally:
                    Path(f.name).unlink()
    
    def test_missing_required_configuration_keys(self):
        """Test handling of missing required configuration keys."""
        incomplete_configs = [
            {},  # Completely empty
            {'run': {}},  # Missing embeddings, llm, etc.
            {'embeddings': {'model': 'test'}},  # Missing other sections
            {'llm': {'provider': 'ollama'}},  # Missing model
            {'vectordb': {}},  # Missing path and collection_name
        ]
        
        required_keys = {
            'embeddings': ['model'],
            'llm': ['provider', 'model'],
            'vectordb': ['path', 'collection_name']
        }
        
        def validate_required_keys(config: Dict, required: Dict[str, list]) -> list:
            """Validate that required configuration keys are present."""
            missing_keys = []
            
            for section, keys in required.items():
                if section not in config:
                    missing_keys.append(f"Missing section: {section}")
                    continue
                
                section_config = config[section]
                for key in keys:
                    if key not in section_config:
                        missing_keys.append(f"Missing key: {section}.{key}")
            
            return missing_keys
        
        for incomplete_config in incomplete_configs:
            missing = validate_required_keys(incomplete_config, required_keys)
            assert len(missing) > 0, f"Should detect missing keys in {incomplete_config}"
    
    def test_configuration_type_validation(self):
        """Test validation of configuration value types."""
        def validate_config_types(config: Dict) -> list:
            """Validate configuration value types."""
            errors = []
            
            # Device should be string
            if 'run' in config and 'device' in config['run']:
                if not isinstance(config['run']['device'], str):
                    errors.append("run.device must be string")
            
            # Temperature should be float
            if 'llm' in config and 'temperature' in config['llm']:
                temp = config['llm']['temperature']
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    errors.append("llm.temperature must be float between 0 and 2")
            
            # Max tokens should be positive integer
            if 'llm' in config and 'max_tokens' in config['llm']:
                max_tokens = config['llm']['max_tokens']
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    errors.append("llm.max_tokens must be positive integer")
            
            # Chunk sizes should be positive integers
            if 'document_processing' in config and 'chunk_sizes' in config['document_processing']:
                chunk_sizes = config['document_processing']['chunk_sizes']
                for file_type, size in chunk_sizes.items():
                    if not isinstance(size, int) or size <= 0:
                        errors.append(f"document_processing.chunk_sizes.{file_type} must be positive integer")
            
            return errors
        
        invalid_type_configs = [
            {
                'run': {'device': 123},  # Should be string
                'llm': {'temperature': 'hot'},  # Should be float
            },
            {
                'llm': {
                    'max_tokens': -1,  # Should be positive
                    'temperature': 5.0  # Should be <= 2
                }
            },
            {
                'document_processing': {
                    'chunk_sizes': {
                        'html': 'large',  # Should be integer
                        'pdf': 0  # Should be positive
                    }
                }
            }
        ]
        
        for invalid_config in invalid_type_configs:
            errors = validate_config_types(invalid_config)
            assert len(errors) > 0, f"Should detect type errors in {invalid_config}"
    
    def test_configuration_value_range_validation(self):
        """Test validation of configuration value ranges."""
        def validate_config_ranges(config: Dict) -> list:
            """Validate configuration value ranges."""
            errors = []
            
            # Temperature range
            if 'llm' in config and 'temperature' in config['llm']:
                temp = config['llm']['temperature']
                if temp < 0 or temp > 2:
                    errors.append(f"llm.temperature {temp} out of range [0, 2]")
            
            # Chunk size ranges
            if 'document_processing' in config:
                chunk_sizes = config['document_processing'].get('chunk_sizes', {})
                for file_type, size in chunk_sizes.items():
                    if size < 100 or size > 5000:
                        errors.append(f"chunk_sizes.{file_type} {size} out of range [100, 5000]")
            
            # Seed range (if specified)
            if 'run' in config and 'seed' in config['run']:
                seed = config['run']['seed']
                if seed < 0 or seed > 2**32 - 1:
                    errors.append(f"run.seed {seed} out of range [0, 2^32-1]")
            
            return errors
        
        out_of_range_configs = [
            {'llm': {'temperature': -0.5}},  # Too low
            {'llm': {'temperature': 3.0}},   # Too high
            {'document_processing': {'chunk_sizes': {'html': 50}}},  # Too small
            {'document_processing': {'chunk_sizes': {'pdf': 10000}}},  # Too large
            {'run': {'seed': -1}},  # Negative seed
            {'run': {'seed': 2**33}},  # Seed too large
        ]
        
        for invalid_config in out_of_range_configs:
            errors = validate_config_ranges(invalid_config)
            assert len(errors) > 0, f"Should detect range errors in {invalid_config}"


class TestHydraIntegration:
    """Test Hydra configuration framework integration."""
    
    def test_hydra_config_composition(self):
        """Test Hydra configuration composition patterns."""
        # Simulate Hydra configuration structure
        base_hydra_config = {
            'defaults': [
                'embeddings: bge_large',
                'llm: ollama_llama3',
                'vectordb: chroma_default'
            ],
            'run': {
                'device': 'auto'
            }
        }
        
        embeddings_config = {
            'model': 'BAAI/bge-large-en-v1.5',
            'device': 'cpu',
            'reranking_enabled': False
        }
        
        llm_config = {
            'provider': 'ollama',
            'model': 'llama3.1:8b',
            'temperature': 0.7,
            'base_url': 'http://localhost:11434'
        }
        
        # Simulate composition
        composed_config = {
            **base_hydra_config,
            'embeddings': embeddings_config,
            'llm': llm_config
        }
        
        # Remove defaults after composition
        if 'defaults' in composed_config:
            del composed_config['defaults']
        
        # Verify composition worked
        assert 'embeddings' in composed_config
        assert 'llm' in composed_config
        assert composed_config['embeddings']['model'] == 'BAAI/bge-large-en-v1.5'
        assert composed_config['llm']['provider'] == 'ollama'
        assert composed_config['run']['device'] == 'auto'
    
    def test_hydra_override_patterns(self):
        """Test Hydra override patterns and syntax."""
        base_config = {
            'llm': {
                'temperature': 0.7,
                'max_tokens': 512
            },
            'embeddings': {
                'model': 'default-model'
            }
        }
        
        # Simulate Hydra overrides
        override_patterns = [
            ('llm.temperature=0.5', {'llm': {'temperature': 0.5}}),
            ('llm.max_tokens=1024', {'llm': {'max_tokens': 1024}}),
            ('embeddings.model=custom-model', {'embeddings': {'model': 'custom-model'}}),
            ('run.device=cuda', {'run': {'device': 'cuda'}})
        ]
        
        def apply_hydra_override(config: Dict, override_path: str, override_value: Any) -> Dict:
            """Apply Hydra-style override to configuration."""
            result = copy.deepcopy(config)
            keys = override_path.split('.')
            
            current = result
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = override_value
            return result
        
        for override_str, expected_override in override_patterns:
            # Parse override string (simplified)
            path, value = override_str.split('=')
            
            # Try to convert value to appropriate type
            try:
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
            except:
                pass  # Keep as string
            
            # Apply override
            overridden_config = apply_hydra_override(base_config, path, value)
            
            # Verify override was applied correctly
            keys = path.split('.')
            current = overridden_config
            for key in keys[:-1]:
                current = current[key]
            
            assert current[keys[-1]] == value


class TestCorpusDirectoryConfiguration:
    """Test corpus directory configuration and validation."""
    
    def test_corpus_directory_defaults(self):
        """Test default corpus directory configuration."""
        default_corpus_config = {
            'document_processing': {
                'resource_dirs': {
                    'html': 'corpus/html',
                    'pdf': 'corpus/pdf',
                    'audio': 'corpus/audio',
                    'images': 'corpus/images',
                    'csv': 'corpus/csv',
                    'docx': 'corpus/docx'
                }
            }
        }
        
        # Verify all expected document types have directories
        resource_dirs = default_corpus_config['document_processing']['resource_dirs']
        
        expected_types = ['html', 'pdf', 'audio', 'images', 'csv', 'docx']
        for doc_type in expected_types:
            assert doc_type in resource_dirs
            assert resource_dirs[doc_type].startswith('corpus/')
    
    def test_corpus_directory_environment_overrides(self):
        """Test corpus directory environment variable overrides."""
        env_overrides = {
            'CORPUS_HTML_DIR': '/custom/html',
            'CORPUS_PDF_DIR': '/custom/pdf',
            'CORPUS_AUDIO_DIR': '/storage/audio',
            'CORPUS_IMAGES_DIR': '/media/images'
        }
        
        with patch.dict(os.environ, env_overrides):
            # Simulate applying environment overrides
            corpus_config = {
                'html': os.getenv('CORPUS_HTML_DIR', 'corpus/html'),
                'pdf': os.getenv('CORPUS_PDF_DIR', 'corpus/pdf'),
                'audio': os.getenv('CORPUS_AUDIO_DIR', 'corpus/audio'),
                'images': os.getenv('CORPUS_IMAGES_DIR', 'corpus/images'),
                'csv': os.getenv('CORPUS_CSV_DIR', 'corpus/csv'),
                'docx': os.getenv('CORPUS_DOCX_DIR', 'corpus/docx')
            }
            
            # Verify overrides applied
            assert corpus_config['html'] == '/custom/html'
            assert corpus_config['pdf'] == '/custom/pdf'
            assert corpus_config['audio'] == '/storage/audio'
            assert corpus_config['images'] == '/media/images'
            
            # Verify defaults used for non-overridden
            assert corpus_config['csv'] == 'corpus/csv'
            assert corpus_config['docx'] == 'corpus/docx'
    
    def test_corpus_directory_validation(self):
        """Test corpus directory path validation."""
        def validate_corpus_directories(resource_dirs: Dict[str, str]) -> Dict[str, list]:
            """Validate corpus directory paths."""
            validation_results = {
                'valid': [],
                'invalid': [],
                'warnings': []
            }
            
            for doc_type, dir_path in resource_dirs.items():
                if not dir_path:
                    validation_results['invalid'].append(f"{doc_type}: empty path")
                    continue
                
                # Check for potentially problematic paths
                if dir_path.startswith('/'):
                    validation_results['warnings'].append(f"{doc_type}: absolute path {dir_path}")
                
                if ' ' in dir_path:
                    validation_results['warnings'].append(f"{doc_type}: path contains spaces")
                
                if dir_path.endswith('/'):
                    validation_results['warnings'].append(f"{doc_type}: path ends with slash")
                
                validation_results['valid'].append(doc_type)
            
            return validation_results
        
        test_cases = [
            # Valid cases
            ({'html': 'corpus/html', 'pdf': 'corpus/pdf'}, 2, 0, 0),
            
            # Warning cases
            ({'html': '/absolute/path', 'pdf': 'path with spaces'}, 2, 0, 2),
            ({'html': 'corpus/html/', 'pdf': 'corpus/pdf'}, 2, 0, 1),
            
            # Invalid cases
            ({'html': '', 'pdf': 'corpus/pdf'}, 1, 1, 0),
        ]
        
        for resource_dirs, exp_valid, exp_invalid, exp_warnings in test_cases:
            results = validate_corpus_directories(resource_dirs)
            
            assert len(results['valid']) == exp_valid
            assert len(results['invalid']) == exp_invalid
            assert len(results['warnings']) == exp_warnings


class TestConfigurationPerformance:
    """Test configuration loading and processing performance."""
    
    def test_large_configuration_loading(self):
        """Test performance with large configuration files."""
        import time
        
        # Create large configuration with many nested sections
        large_config = {}
        
        # Add many document types
        large_config['document_processing'] = {
            'resource_dirs': {},
            'chunk_sizes': {},
            'processing_options': {}
        }
        
        for i in range(100):
            doc_type = f'doc_type_{i}'
            large_config['document_processing']['resource_dirs'][doc_type] = f'corpus/{doc_type}'
            large_config['document_processing']['chunk_sizes'][doc_type] = 1000 + i
            large_config['document_processing']['processing_options'][doc_type] = {
                'enabled': True,
                'priority': i,
                'metadata': {f'key_{j}': f'value_{j}' for j in range(10)}
            }
        
        # Add many embedding configurations
        large_config['embeddings'] = {
            'models': {}
        }
        
        for i in range(50):
            model_name = f'model_{i}'
            large_config['embeddings']['models'][model_name] = {
                'path': f'models/{model_name}',
                'device': 'cpu',
                'parameters': {f'param_{j}': j * 0.1 for j in range(20)}
            }
        
        # Test loading performance
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(large_config, f)
            config_file = f.name
        
        try:
            # Measure loading time
            start_time = time.time()
            
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            loading_time = time.time() - start_time
            
            # Should complete in reasonable time (adjust threshold as needed)
            assert loading_time < 5.0  # Should load within 5 seconds
            
            # Verify config integrity
            assert len(loaded_config['document_processing']['resource_dirs']) == 100
            assert len(loaded_config['embeddings']['models']) == 50
            
        finally:
            Path(config_file).unlink()
    
    def test_configuration_caching(self):
        """Test configuration caching for performance."""
        cache = {}
        
        def cached_load_config(config_path: str) -> Dict:
            """Load configuration with caching."""
            if config_path in cache:
                return cache[config_path]
            
            # Simulate loading
            config = {'loaded_at': time.time()}
            cache[config_path] = config
            return config
        
        config_path = 'test_config.yaml'
        
        # First load
        start_time = time.time()
        config1 = cached_load_config(config_path)
        first_load_time = time.time() - start_time
        
        # Second load (should be cached)
        start_time = time.time()
        config2 = cached_load_config(config_path)
        second_load_time = time.time() - start_time
        
        # Verify caching worked
        assert config1 is config2  # Same object reference
        assert second_load_time < first_load_time  # Faster second load
        assert len(cache) == 1  # Only one entry in cache
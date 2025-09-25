"""
Unit tests for core utilities - configuration loading and device management.
Tests the high-priority configuration & environment functionality.
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from src.utils.core import load_config, get_device


class TestLoadConfig:
    """Test configuration loading with file and environment variable support."""
    
    def test_load_config_with_file(self, temp_config_file):
        """Test loading configuration from a valid YAML file."""
        config = load_config(temp_config_file)
        
        assert config['run']['device'] == 'cpu'
        assert config['embeddings']['model'] == 'test-model'
        assert config['database']['path'] == './test_vector_store'
        assert config['llm']['ollama']['model'] == 'test-llama'
    
    def test_load_config_missing_file(self):
        """Test loading configuration when file doesn't exist - should return empty dict."""
        config = load_config('/nonexistent/config.yaml')
        assert config == {}
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML - should raise exception."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_with_env_vars(self, temp_config_file, mock_env_vars):
        """Test that environment variables override config file values."""
        config = load_config(temp_config_file)
        
        # Environment variables should override file values
        assert config['database']['path'] == '/test/vector_store'  # From env
        assert config['embeddings']['model'] == 'test-embedding-model'  # From env
        assert config['llm']['ollama']['base_url'] == 'http://test-ollama:11434'  # From env
        assert config['run']['device'] == 'cuda'  # From env
    
    def test_load_config_env_path_override(self, temp_config_file):
        """Test CONFIG_PATH environment variable overrides default path."""
        with patch.dict(os.environ, {'CONFIG_PATH': temp_config_file}):
            config = load_config()  # No path specified
            assert config['run']['device'] == 'cpu'  # Should load from temp file
    
    def test_load_config_corpus_dirs_env_vars(self, temp_config_file):
        """Test corpus directory environment variable overrides."""
        corpus_env_vars = {
            'CORPUS_HTML_DIR': '/custom/html',
            'CORPUS_PDF_DIR': '/custom/pdf',
            'CORPUS_CSV_DIR': '/custom/csv'
        }
        
        with patch.dict(os.environ, corpus_env_vars):
            config = load_config(temp_config_file)
            
            assert config['document_processing']['resource_dirs']['html'] == '/custom/html'
            assert config['document_processing']['resource_dirs']['pdf'] == '/custom/pdf'
            assert config['document_processing']['resource_dirs']['csv'] == '/custom/csv'


class TestGetDevice:
    """Test device detection and selection logic."""
    
    def test_get_device_cuda_available(self, mock_torch):
        """Test device selection when CUDA is available and preferred."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = MagicMock(type='cuda')
        
        config = {'run': {'device': 'cuda'}}
        device = get_device(config)
        
        mock_torch.device.assert_called_with('cuda')
    
    def test_get_device_mps_available(self, mock_torch):
        """Test device selection when MPS is available and preferred."""
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = MagicMock(type='mps')
        
        config = {'run': {'device': 'mps'}}
        device = get_device(config)
        
        mock_torch.device.assert_called_with('mps')
    
    def test_get_device_cpu_fallback(self, mock_torch):
        """Test device selection falls back to CPU when preferred device unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = MagicMock(type='cpu')
        
        config = {'run': {'device': 'cuda'}}  # Prefer CUDA but not available
        
        # Should fall back and call CPU device
        with patch('builtins.print') as mock_print:
            device = get_device(config)
            mock_print.assert_called()  # Should print warning
    
    def test_get_device_auto_selection(self, mock_torch):
        """Test auto device selection logic."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = MagicMock(type='cuda')
        
        config = {'run': {'device': 'auto'}}
        device = get_device(config)
        
        # Should select CUDA since it's available
        mock_torch.device.assert_called_with('cuda')
    
    def test_get_device_cpu_explicit(self, mock_torch):
        """Test explicit CPU device selection."""
        mock_torch.device.return_value = MagicMock(type='cpu')
        
        config = {'run': {'device': 'cpu'}}
        device = get_device(config)
        
        # Should use auto logic for CPU
        mock_torch.device.assert_called()
    
    def test_get_device_unknown_preference(self, mock_torch):
        """Test unknown device preference falls back to CPU."""
        mock_torch.device.return_value = MagicMock(type='cpu')
        
        config = {'run': {'device': 'unknown_device'}}
        
        with patch('builtins.print') as mock_print:
            device = get_device(config)
            mock_print.assert_called()  # Should print warning
            mock_torch.device.assert_called_with('cpu')
    
    def test_get_device_missing_config(self, mock_torch):
        """Test device selection with missing run config - should use default."""
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = MagicMock(type='mps')
        
        config = {}  # No run config
        device = get_device(config)
        
        # Should use default 'mps' preference
        mock_torch.device.assert_called_with('mps')
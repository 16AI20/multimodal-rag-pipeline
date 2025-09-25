"""
Utilities package for the RAG pipeline.
Contains core utilities, content safety, and UI helper functions.
"""

# Core utilities (configuration, device management, etc.)
from .core import (
    load_config, get_device, setup_logging, preprocess_query, DocumentChunker
)

# Content safety and moderation
from .content_safety import (
    check_input_safety, check_output_safety, enhanced_content_analysis
)

# UI helper functions
from .ui_helpers import (
    format_confidence_indicator, add_contextual_disclaimer, format_safety_warnings
)

__all__ = [
    # Core utilities
    'load_config',
    'get_device', 
    'setup_logging',
    'preprocess_query',
    'DocumentChunker',
    
    # Content safety
    'check_input_safety',
    'check_output_safety',
    'enhanced_content_analysis',
    
    # UI helpers
    'format_confidence_indicator',
    'add_contextual_disclaimer', 
    'format_safety_warnings'
]
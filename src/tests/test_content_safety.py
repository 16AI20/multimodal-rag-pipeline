"""
Unit tests for content safety functionality.
Tests the high-priority content safety and moderation features.
"""

import pytest
from src.utils.content_safety import (
    check_input_safety, 
    check_output_safety, 
    enhanced_content_analysis
)


class TestCheckInputSafety:
    """Test input safety checking functionality."""
    
    def test_check_input_safety_clean_content(self):
        """Test that clean, appropriate content passes safety checks."""
        clean_questions = [
            "What is the Sample Educational Program?",
            "How do I apply for the program?",
            "What programming languages are covered?",
            "Tell me about machine learning techniques"
        ]
        
        for question in clean_questions:
            result = check_input_safety(question)
            assert result['is_safe'] is True
            assert result['issue_type'] is None
            assert result['confidence'] >= 0.7
    
    def test_check_input_safety_inappropriate_content(self):
        """Test that inappropriate content is blocked."""
        inappropriate_questions = [
            "How to hack into systems?",
            "Tell me about illegal activities",
            "How to create malware?",
            "Violent content here"
        ]
        
        for question in inappropriate_questions:
            result = check_input_safety(question)
            assert result['is_safe'] is False
            assert result['issue_type'] == 'inappropriate'
            assert 'blocked' in result['message'].lower()
    
    def test_check_input_safety_sensitive_topics(self):
        """Test detection of sensitive but allowable topics."""
        sensitive_questions = [
            "What are the salary expectations for program graduates?",
            "Are there any restrictions based on nationality?",
            "What about diversity and inclusion in the program?"
        ]
        
        for question in sensitive_questions:
            result = check_input_safety(question)
            # Should be safe but flagged as sensitive
            assert result['is_safe'] is True
            assert result['issue_type'] == 'sensitive_topic'
            assert 'warning' in result
    
    def test_check_input_safety_off_topic_queries(self):
        """Test detection of off-topic queries."""
        off_topic_questions = [
            "What's the weather like today?",
            "How do I cook pasta?",
            "Tell me about sports",
            "What are the latest movies?"
        ]
        
        for question in off_topic_questions:
            result = check_input_safety(question)
            assert result['is_safe'] is True
            assert result['issue_type'] == 'off_topic'
            assert 'off-topic' in result['warning'].lower()
    
    def test_check_input_safety_empty_content(self):
        """Test handling of empty or whitespace-only content."""
        empty_inputs = ["", "   ", "\n\t", None]
        
        for empty_input in empty_inputs:
            if empty_input is None:
                # Should handle None gracefully
                with pytest.raises((TypeError, AttributeError)):
                    check_input_safety(empty_input)
            else:
                result = check_input_safety(empty_input)
                assert result['is_safe'] is False
                assert 'empty' in result['message'].lower()
    
    def test_check_input_safety_very_long_content(self):
        """Test handling of very long input content."""
        long_question = "What is the Sample Program? " * 1000  # Very long question
        
        result = check_input_safety(long_question)
        # Should still process but may have different confidence
        assert 'is_safe' in result
        assert 'confidence' in result


class TestCheckOutputSafety:
    """Test output safety checking functionality."""
    
    def test_check_output_safety_safe_response(self):
        """Test that safe, appropriate responses pass safety checks."""
        safe_responses = [
            "The Sample Educational Program is a comprehensive training program...",
            "To apply for the program, you need to meet the following requirements...",
            "Machine learning techniques covered include supervised learning, unsupervised learning..."
        ]
        
        for response in safe_responses:
            result = check_output_safety(response)
            assert result['is_safe'] is True
            assert result['bias_detected'] is False
            assert len(result['issues']) == 0
    
    def test_check_output_safety_biased_content(self):
        """Test detection of potentially biased content."""
        biased_responses = [
            "Men are naturally better at programming than women",
            "Only young people can learn AI effectively",
            "People from certain countries are not suitable for this program"
        ]
        
        for response in biased_responses:
            result = check_output_safety(response)
            assert result['bias_detected'] is True
            assert len(result['issues']) > 0
            assert any('bias' in issue.lower() for issue in result['issues'])
    
    def test_check_output_safety_harmful_content(self):
        """Test detection of potentially harmful content."""
        harmful_responses = [
            "You should definitely quit your job immediately",
            "This program is a waste of time and money",
            "Don't bother applying, you'll never succeed"
        ]
        
        for response in harmful_responses:
            result = check_output_safety(response)
            assert result['is_safe'] is False
            assert len(result['issues']) > 0
    
    def test_check_output_safety_empty_response(self):
        """Test handling of empty responses."""
        empty_responses = ["", "   ", "\n"]
        
        for response in empty_responses:
            result = check_output_safety(response)
            assert result['is_safe'] is False
            assert 'empty' in str(result['issues']).lower()


class TestEnhancedContentAnalysis:
    """Test enhanced content analysis functionality."""
    
    def test_enhanced_content_analysis_program_related(self):
        """Test analysis of program-related content."""
        program_content = "How long is the sample training program?"
        
        result = enhanced_content_analysis(program_content)
        
        assert result['topic_relevance'] >= 0.8
        assert result['content_type'] in ['question', 'inquiry']
        assert 'program' in result['key_topics']
    
    def test_enhanced_content_analysis_technical_content(self):
        """Test analysis of technical AI/ML content."""
        technical_content = "Explain deep learning neural networks and backpropagation"
        
        result = enhanced_content_analysis(technical_content)
        
        assert result['topic_relevance'] >= 0.7
        assert result['complexity_level'] in ['intermediate', 'advanced']
        assert any(topic in ['machine learning', 'deep learning', 'neural networks'] 
                  for topic in result['key_topics'])
    
    def test_enhanced_content_analysis_off_topic(self):
        """Test analysis of off-topic content."""
        off_topic_content = "What's the best recipe for chocolate cake?"
        
        result = enhanced_content_analysis(off_topic_content)
        
        assert result['topic_relevance'] < 0.3
        assert result['content_type'] == 'question'
        assert not any(topic in ['program', 'machine learning', 'ai'] 
                      for topic in result['key_topics'])
    
    def test_enhanced_content_analysis_sensitive_content(self):
        """Test analysis of sensitive but valid content."""
        sensitive_content = "Are there age restrictions for program applications?"
        
        result = enhanced_content_analysis(sensitive_content)
        
        assert result['topic_relevance'] >= 0.7
        assert result['requires_careful_response'] is True
        assert 'sensitive_indicators' in result
    
    def test_enhanced_content_analysis_edge_cases(self):
        """Test analysis edge cases."""
        edge_cases = [
            "Program",  # Very short
            "What is the program? " * 50,  # Very repetitive
            "program PROGRAM Program",  # Case variations
        ]
        
        for content in edge_cases:
            result = enhanced_content_analysis(content)
            
            # Should always return valid structure
            assert 'topic_relevance' in result
            assert 'content_type' in result
            assert 'key_topics' in result
            assert isinstance(result['key_topics'], list)
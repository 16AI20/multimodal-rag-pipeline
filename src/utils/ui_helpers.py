"""
UI utility functions for Streamlit interface.
Handles confidence indicators, disclaimers, and response formatting.
"""

from typing import Dict, Any


def format_confidence_indicator(retrieval_metadata: Dict[str, Any]) -> str:
    """
    Format confidence and quality indicators for display.
    
    Args:
        retrieval_metadata: Metadata from retrieval system
        
    Returns:
        Formatted warning string for low confidence cases
    """
    if not retrieval_metadata:
        return ""
    
    confidence_level = retrieval_metadata.get('confidence_level', 'unknown')
    
    # Only add uncertainty warning for low confidence (no confidence bar since it's in Search Analysis)
    warning = ""
    if confidence_level in ['low', 'very_low']:
        warning = "\n\n⚠️ *Low confidence - please verify this information with official sources*"
    
    return warning


def add_contextual_disclaimer(question: str, response: str) -> str:
    """
    Add contextual disclaimers based on question content.
    
    Args:
        question: User's question
        response: AI response
        
    Returns:
        Response with appropriate disclaimer added
    """
    # Keywords that might require professional advice disclaimers
    professional_keywords = [
        "apply", "application", "deadline", "admission", "accept", "reject",
        "career", "job", "salary", "worth it", "should i", "recommend",
        "decide", "choose", "better", "advice"
    ]
    
    # Keywords that might require accuracy disclaimers
    factual_keywords = [
        "when", "date", "cost", "fee", "price", "requirement", "eligibility",
        "how long", "duration", "location", "contact"
    ]
    
    question_lower = question.lower()
    
    # Check for professional advice questions
    if any(keyword in question_lower for keyword in professional_keywords):
        response += "\n\n💡 *For official guidance on applications and important decisions, please consult relevant authorities or original sources.*"
    
    # Check for factual information questions
    elif any(keyword in question_lower for keyword in factual_keywords):
        response += "\n\n💡 *Please verify current details with official sources as information may change.*"
    
    return response


def format_safety_warnings(output_safety: Dict[str, Any]) -> str:
    """
    Format safety warnings for output issues.
    
    Args:
        output_safety: Safety check results
        
    Returns:
        Formatted safety warning string
    """
    if output_safety.get("is_safe", True):
        return ""
    
    warnings = ""
    for issue in output_safety.get("issues", []):
        if issue["type"] == "potential_bias":
            warnings += "\n\n⚠️ *This response may contain bias. Please consider multiple perspectives.*"
        elif issue["type"] == "non_inclusive_language":
            warnings += "\n\n💡 *Note: Some terms could be more inclusive. Consider alternative phrasing.*"
    
    return warnings
"""
Content safety and moderation utilities.
Provides enhanced rule-based content filtering with context analysis and intent detection.
"""

import re
from typing import Dict, Any


def enhanced_content_analysis(text: str) -> Dict[str, Any]:
    """
    Enhanced rule-based content analysis with context awareness.
    Analyzes intent patterns, context, and linguistic indicators.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    # 1. Intent Pattern Analysis (regex-based)
    harmful_intent_patterns = [
        # Instruction-seeking patterns
        (r'\b(?:how to|teach me|show me|help me)\s+(?:hack|exploit|bypass|break)\b', 
         'instruction_seeking', 'high'),
        
        # Information extraction patterns  
        (r'\b(?:get|find|obtain)\s+(?:personal|private|confidential)\s+(?:info|data|details)\b',
         'information_extraction', 'high'),
         
        # Manipulation patterns
        (r'\b(?:manipulate|trick|fool|deceive)\b',
         'manipulation', 'medium'),
         
        # System testing patterns
        (r'\b(?:test|try|attempt)\s+(?:security|limits|boundaries)\b',
         'system_testing', 'medium')
    ]
    
    # 2. Context Analysis for risky terms
    def analyze_context(word: str, context_window: int = 3) -> str:
        """Check if risky word is used in harmful context."""
        words = text.lower().split()
        if word not in ' '.join(words):
            return 'safe'
            
        word_positions = [i for i, w in enumerate(words) if word in w]
        
        for pos in word_positions:
            start = max(0, pos - context_window)
            end = min(len(words), pos + context_window + 1)
            context = ' '.join(words[start:end])
            
            # Harmful context indicators
            harmful_indicators = ['how to', 'teach me', 'help me', 'show me', 'illegal', 'against']
            # Safe context indicators  
            safe_indicators = ['hack together', 'life hack', 'growth hack', 'hack day']
            
            # Check for safe contexts first
            if any(safe in context for safe in safe_indicators):
                return 'safe'
            # Then check for harmful contexts
            elif any(harmful in context for harmful in harmful_indicators):
                return 'harmful'
                
        return 'neutral'
    
    # 3. Topic Relevance Scoring (Domain-Agnostic)
    def calculate_relevance_score(text: str) -> float:
        """Calculate how relevant the question is to document corpus topics."""
        # Generic educational and informational terms
        relevant_terms = {
            'educational': ['learn', 'study', 'course', 'training', 'tutorial', 'guide'],
            'informational': ['what', 'how', 'why', 'when', 'where', 'explain', 'describe'],
            'technical': ['system', 'process', 'method', 'approach', 'technique', 'implementation'],
            'contextual': ['help', 'information', 'details', 'documentation', 'reference']
        }
        
        text_lower = text.lower()
        score = 0
        
        # Weight terms by category relevance
        for term in relevant_terms['educational']:
            if term in text_lower:
                score += 3
        for term in relevant_terms['informational']:
            if term in text_lower:
                score += 2
        for term in relevant_terms['technical']:
            if term in text_lower:
                score += 2
        for term in relevant_terms['contextual']:
            if term in text_lower:
                score += 1
                
        return min(score, 15) / 15.0  # Normalize to 0-1
    
    # Run analysis
    text_lower = text.lower()
    
    # Pattern matching
    detected_patterns = []
    for pattern, intent_type, severity in harmful_intent_patterns:
        if re.search(pattern, text_lower):
            detected_patterns.append({
                'intent': intent_type,
                'severity': severity
            })
    
    # Context analysis for risky terms
    context_risks = []
    risky_terms = ['hack', 'exploit', 'bypass', 'illegal', 'personal']
    for term in risky_terms:
        context_result = analyze_context(term)
        if context_result == 'harmful':
            context_risks.append(term)
    
    # Relevance scoring
    relevance_score = calculate_relevance_score(text)
    
    return {
        'detected_patterns': detected_patterns,
        'context_risks': context_risks,
        'relevance_score': relevance_score
    }


def check_input_safety(question: str) -> Dict[str, Any]:
    """
    Enhanced input safety check with multi-layer analysis.
    Combines pattern detection, context analysis, and relevance scoring.
    
    Args:
        question: User input to check
        
    Returns:
        Dictionary with safety assessment and blocking/warning information
    """
    # Run enhanced analysis
    analysis = enhanced_content_analysis(question)
    
    # Decision logic based on analysis results
    detected_patterns = analysis['detected_patterns']
    context_risks = analysis['context_risks']
    relevance_score = analysis['relevance_score']
    
    # High-risk patterns detected
    high_risk_patterns = [p for p in detected_patterns if p['severity'] == 'high']
    if high_risk_patterns:
        intents = [p['intent'] for p in high_risk_patterns]
        
        # Create user-friendly messages for different intent types
        intent_messages = {
            'instruction_seeking': 'harmful or inappropriate instructions',
            'information_extraction': 'accessing private or confidential information',
            'manipulation': 'manipulation or deception techniques',
            'system_testing': 'testing security boundaries'
        }
        
        # Convert technical intent names to user-friendly descriptions
        friendly_descriptions = [intent_messages.get(intent, intent) for intent in intents]
        
        return {
            "is_safe": False,
            "issue_type": "inappropriate_content",
            "detected_patterns": intents,
            "message": f"I cannot assist with requests involving {', '.join(friendly_descriptions)}. Please ask educational or informational questions instead."
        }
    
    # Context-based risks
    if context_risks:
        return {
            "is_safe": False,
            "issue_type": "inappropriate_context",
            "blocked_keywords": context_risks,
            "message": f"The terms '{', '.join(context_risks)}' appear to be used in an inappropriate context. Please rephrase your question."
        }
    
    # Medium-risk patterns with low relevance
    medium_risk_patterns = [p for p in detected_patterns if p['severity'] == 'medium']
    if medium_risk_patterns and relevance_score < 0.3:
        return {
            "is_safe": False,
            "issue_type": "suspicious_intent",
            "detected_patterns": [p['intent'] for p in medium_risk_patterns],
            "message": "This request appears to have concerning intent and is not related to the document corpus. Please ask educational or informational questions."
        }
    
    # Sensitive topics check (original logic for now)
    sensitive_topics = ['salary', 'income', 'personal', 'private', 'confidential',
                       'medical', 'health', 'diagnosis', 'treatment',
                       'legal', 'lawsuit', 'contract', 'rights',
                       'invest', 'loan', 'mortgage', 'financial advice']
    
    question_lower = question.lower()
    sensitive_found = [kw for kw in sensitive_topics if kw in question_lower]
    if sensitive_found:
        return {
            "is_safe": True,
            "issue_type": "sensitive_topic",
            "sensitive_keywords": sensitive_found,
            "warning": "This appears to be a sensitive topic. Please remember that I provide educational information only and cannot give professional advice."
        }
    
    # Off-topic with very low relevance
    if relevance_score < 0.1 and len(question.split()) > 3:
        return {
            "is_safe": True,
            "issue_type": "off_topic",
            "relevance_score": relevance_score,
            "warning": "This question seems unrelated to the document corpus. Could you rephrase to focus on topics covered in the available documents?"
        }
    
    return {
        "is_safe": True,
        "issue_type": None,
        "analysis": analysis  # Include analysis results for transparency
    }


def check_output_safety(response: str) -> Dict[str, Any]:
    """
    Check if AI response contains inappropriate content or bias.
    
    Args:
        response: AI-generated response
        
    Returns:
        Dictionary with safety assessment and any corrections needed
    """
    # Bias detection keywords
    bias_indicators = [
        # Gender bias
        'men are', 'women are', 'guys', 'girls',
        # Age bias  
        'young people', 'older people', 'millennials', 'boomers',
        # Educational bias
        'smart people', 'dumb', 'stupid', 'intelligent people',
        # Absolute statements that could be biased
        'always', 'never', 'all', 'none', 'everyone', 'no one'
    ]
    
    # Inclusive language issues
    non_inclusive_terms = {
        'guys': 'everyone/folks',
        'manpower': 'workforce/staff',
        'whitelist': 'allowlist',
        'blacklist': 'blocklist',
        'master': 'primary/main',
        'slave': 'secondary/replica'
    }
    
    response_lower = response.lower()
    
    # Check for bias indicators
    bias_found = [bias for bias in bias_indicators if bias in response_lower]
    
    # Check for non-inclusive language
    inclusive_issues = {term: replacement for term, replacement in non_inclusive_terms.items() 
                       if term in response_lower}
    
    issues = []
    if bias_found:
        issues.append({
            "type": "potential_bias",
            "indicators": bias_found,
            "severity": "medium"
        })
    
    if inclusive_issues:
        issues.append({
            "type": "non_inclusive_language", 
            "terms": inclusive_issues,
            "severity": "low"
        })
    
    return {
        "is_safe": len(issues) == 0,
        "issues": issues,
        "needs_review": len([i for i in issues if i["severity"] == "medium"]) > 0
    }
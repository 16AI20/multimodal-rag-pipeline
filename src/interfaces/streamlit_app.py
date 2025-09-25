#!/usr/bin/env python
"""
Enhanced Streamlit frontend for RAG pipeline.
Provides a modern, aesthetically pleasing chatbot interface for querying document corpus.

Usage:
    streamlit run src/interfaces/streamlit_app.py
"""

import os
import sys
from pathlib import Path

# Set environment variables before importing other libraries
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Ensure project root is in Python path for absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from typing import Dict, Any, List
from datetime import datetime
import json
import requests
import time

# Import our modular components
from src.utils import (
    check_input_safety, check_output_safety,
    format_confidence_indicator, add_contextual_disclaimer, format_safety_warnings
)


# FastAPI backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def check_backend_health() -> Dict[str, Any]:
    """Check if FastAPI backend is available.
    
    Returns:
        Dictionary containing backend health status and any error messages.
    """
    try:
        # First try a simple root endpoint which should be faster
        response = requests.get(f"{BACKEND_URL}/", timeout=10)
        if response.status_code == 200:
            # If root works, try the health endpoint with longer timeout
            health_response = requests.get(f"{BACKEND_URL}/health", timeout=30)
            if health_response.status_code == 200:
                return health_response.json()
            else:
                # Root works but health check fails - backend is partially ready
                return {"status": "initializing", "error": f"Backend initializing (health check returned {health_response.status_code})"}
        else:
            return {"status": "unhealthy", "error": f"Backend returned {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "unhealthy", "error": "Backend is starting up or overloaded (timeout). Please wait and try again."}
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": f"Cannot connect to backend: {str(e)}"}

@st.cache_data(ttl=30)  # Cache for only 30 seconds
def get_collection_info() -> Dict[str, Any]:
    """Get collection information from FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/collection", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Backend unavailable (status {response.status_code})"}
    except requests.exceptions.RequestException:
        return {"error": "Backend not running"}

def query_rag_backend(question: str, k: int = 5, file_type: str = None, stream: bool = False) -> Dict[str, Any]:
    """Query the RAG system via FastAPI backend.
    
    Args:
        question: User's question to ask.
        k: Number of documents to retrieve.
        file_type: Optional file type filter.
        stream: Whether to use streaming response.
        
    Returns:
        Response dictionary or payload for streaming.
    """
    try:
        payload = {
            "question": question,
            "k": k,
            "file_type": file_type,  # None will be serialized as null in JSON
            "return_sources": True,
            "stream": stream
        }
        
        if stream:
            # For streaming, we handle the response differently
            return payload  # Return payload for streaming handler
        else:
            response = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Backend returned {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Cannot connect to backend: {str(e)}"}


def stream_rag_response(payload: Dict[str, Any], message_placeholder) -> Dict[str, Any]:
    """Stream RAG response using Server-Sent Events from FastAPI backend.
    
    Args:
        payload: Request payload for streaming.
        message_placeholder: Streamlit placeholder for displaying streaming response.
        
    Returns:
        Complete response dictionary after streaming.
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Backend returned {response.status_code}: {response.text}"}
        
        full_answer = ""
        sources = []
        retrieval_info = None
        retrieval_metadata = {}
        complete_response = None
        first_token_received = False
        
        # Process streaming response
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    # Parse the JSON data
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    chunk_type = data.get("type")
                    
                    if chunk_type == "error":
                        return {"error": data.get('error', 'Unknown streaming error')}
                        
                    elif chunk_type == "sources":
                        sources = data.get("sources", [])
                        retrieval_info = data.get("retrieval_info", {})
                        retrieval_metadata = data.get("retrieval_metadata", {})
                        # Sources will be displayed in sidebar after completion
                    
                    elif chunk_type == "token":
                        token = data.get("content", "")
                        if token:
                            if not first_token_received:
                                first_token_received = True
                                # Clear any loading message and start showing content
                                message_placeholder.markdown("...")
                            
                            full_answer += token
                            # Update message display with cursor
                            message_placeholder.markdown(full_answer + "▌")
                    
                    elif chunk_type == "complete":
                        complete_response = data.get("complete_response", {})
                        # Remove cursor from final message
                        message_placeholder.markdown(full_answer)
                        break
                        
                except json.JSONDecodeError as e:
                    st.error(f"JSON decode error: {e}")
                    continue
        
        if complete_response:
            return {
                "answer": complete_response.get("answer", full_answer),
                "sources": sources,
                "retrieval_info": retrieval_info,
                "retrieval_metadata": retrieval_metadata
            }
        else:
            return {"error": "Streaming completed without final response"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Streaming connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Streaming error: {str(e)}"}



def initialize_session_state() -> None:
    """Initialize session state variables for the Streamlit app."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    
    if "current_metadata" not in st.session_state:
        st.session_state.current_metadata = {}
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "backend_health_checked" not in st.session_state:
        st.session_state.backend_health_checked = False
    if "backend_health" not in st.session_state:
        st.session_state.backend_health = {"status": "unknown"}


def render_chat_message(message: Dict[str, Any], is_user: bool = False) -> None:
    """Render a single chat message in ChatGPT/Claude style.
    
    Args:
        message: Message dictionary containing content and metadata.
        is_user: Whether this is a user message or assistant message.
    """
    timestamp = message.get("timestamp", "")
    content = message["content"]
    
    if is_user:
        # User message with avatar and natural flow
        with st.chat_message("user"):
            st.markdown(content)
    else:
        # Assistant message with avatar and natural flow
        with st.chat_message("assistant"):
            st.markdown(content)


def render_sources_panel(sources: List[Dict[str, Any]], retrieval_metadata: Dict[str, Any] = None) -> None:
    """Render the sources panel with enhanced styling and metadata.
    
    Args:
        sources: List of source documents with metadata.
        retrieval_metadata: Optional retrieval confidence and quality metrics.
    """
    
    # Display retrieval metadata if available
    if retrieval_metadata:
        st.subheader("🔍 Search Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence = retrieval_metadata.get('confidence_score', 0)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col2:
            quality = retrieval_metadata.get('retrieval_quality', 'unknown')
            st.metric("Quality", quality.title())
        
        with col3:
            unique_sources = retrieval_metadata.get('unique_sources', 0)
            st.metric("Unique Sources", unique_sources)
        
        # Additional details in expander
        with st.expander("📊 Detailed Search Metrics"):
            st.write(f"**Best Match Score**: {retrieval_metadata.get('best_match_score', 0):.1%}")
            st.write(f"**Average Match Score**: {retrieval_metadata.get('avg_match_score', 0):.1%}")
            st.write(f"**Documents Found**: {retrieval_metadata.get('num_documents_found', 0)}")
            st.write(f"**Source Diversity**: {'✅ High' if retrieval_metadata.get('diverse_sources') else '⚠️ Limited'}")
            
            file_types = retrieval_metadata.get('file_types_found', [])
            if file_types:
                st.write(f"**File Types**: {', '.join(file_types).upper()}")
    
    if not sources:
        st.info("🔍 Sources will appear here after you ask a question")
        return
    
    st.subheader("📚 Retrieved Sources")
    
    for i, source in enumerate(sources, 1):
        # Show source quality if available
        quality_indicator = ""
        if retrieval_metadata and 'score_distribution' in retrieval_metadata:
            scores = retrieval_metadata['score_distribution']
            if i <= len(scores):
                score = scores[i-1]
                if score >= 0.8:
                    quality_indicator = " 🟢"
                elif score >= 0.6:
                    quality_indicator = " 🟡"
                else:
                    quality_indicator = " 🟠"
        
        
        with st.expander(f"📄 Source {i}: {source.get('citation', 'Unknown')} ({source.get('file_type', 'unknown').upper()}){quality_indicator}"):
            st.write("**Content Preview:**")
            st.info(source.get('content_preview', 'No preview available'))


def display_collection_stats(info: Dict[str, Any]) -> None:
    """Display collection statistics in sidebar with enhanced styling.
    
    Args:
        info: Collection information dictionary.
    """
    st.sidebar.subheader("📊 Collection Statistics")
    
    if 'error' in info:
        if info['error'] == "Backend not running":
            st.sidebar.warning("⚠️ Backend not running")
            st.sidebar.info("💡 Start backend: `./run.sh --mode backend`")
        else:
            st.sidebar.error(f"Error: {info['error']}")
        return
    
    # Display metrics in a nice format
    total_docs = info.get('total_documents', 0)
    file_types = info.get('file_types', [])
    
    # Create columns for metrics
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Documents", total_docs)
    
    with col2:
        st.metric("File Types", len(file_types))
    
    # File types list with counts
    if file_types:
        st.sidebar.write("**Available Types:**")
        file_type_counts = info.get('file_type_counts', {})
        for ft in file_types:
            count = file_type_counts.get(ft, 0)
            st.sidebar.write(f"• {ft.upper()} ({count})")




def export_chat_history() -> None:
    """Export chat history as JSON download."""
    if st.session_state.messages:
        chat_data = {
            "conversation_id": st.session_state.conversation_id,
            "exported_at": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "total_messages": len(st.session_state.messages)
        }
        
        st.sidebar.download_button(
            "📥 Export Chat",
            data=json.dumps(chat_data, indent=2),
            file_name=f"rag_chat_{st.session_state.conversation_id}.json",
            mime="application/json"
        )


def main() -> None:
    """Main Streamlit application with enhanced UI."""
    st.set_page_config(
        page_title="RAG Pipeline Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header with enhanced title
    st.title("🤖 RAG Pipeline Assistant")
    st.markdown("**Ask questions about your documents and knowledge base**")
    
    # Professional Boundaries & Model Limitations Disclaimers
    with st.expander("⚠️ Important Disclaimers - Please Read", expanded=False):
        st.markdown("""
        ### 🎓 Educational Purpose Only
        This chatbot provides **information** based on your uploaded document corpus. 
        It is **not a substitute for**:
        - Official institutional advice or decisions
        - Professional career counseling or guidance
        - Financial or investment advice
        - Technical support for external applications or systems
        
        ### 🤖 AI System Limitations
        Please be aware that this AI assistant:
        - **May contain errors** - Always verify important information with official sources
        - **Has knowledge limitations** - Training data may not include the most recent updates
        - **Cannot access real-time data** - Cannot check current application status, deadlines, or availability
        - **May misinterpret questions** - Complex or ambiguous queries might yield inaccurate responses
        - **Cannot make decisions for you** - This system provides information, not recommendations
        
        ### 📋 For Official Information
        **Always consult official sources for**:
        - Application procedures and deadlines
        - Admission requirements and criteria  
        - Program updates and changes
        - Technical support and assistance
        
        **Verify with original sources when making important decisions**
        """)
    
    st.divider()
    
    # Check backend health only once per session
    if "backend_health_checked" not in st.session_state:
        with st.spinner("🔍 Checking backend connection..."):
            health = check_backend_health()
        
        st.session_state.backend_health_checked = True
        st.session_state.backend_health = health
        
        if health["status"] == "healthy":
            st.success("✅ Backend connected!")
        elif health["status"] == "initializing":
            st.warning(f"⏳ Backend is initializing: {health.get('error', 'Please wait...')}")
            st.info("💡 The backend is loading models. This may take a minute. You can still proceed.")
        else:
            st.error(f"❌ Backend connection failed: {health.get('error', 'Unknown error')}")
            st.error("💡 Make sure FastAPI backend is running: `./run.sh --mode backend`")
            st.stop()
    else:
        # Show cached health status
        health = st.session_state.backend_health
        if health["status"] == "healthy":
            st.success("✅ Backend connected!")
        elif health["status"] == "initializing":
            st.warning("⏳ Backend is initializing...")
            st.info("💡 You can still use the chatbot while models are loading.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display collection info
        info = get_collection_info()
        display_collection_stats(info)
        
        
        st.divider()
        
        # Query settings
        st.subheader("🔧 Query Settings")
        
        num_docs = st.slider("📄 Documents to retrieve", 1, 10, 5)
        
        file_type_filter = st.selectbox(
            "📁 Filter by file type",
            options=[None, "html", "pdf", "docx", "csv", "audio", "image"],
            format_func=lambda x: "All types" if x is None else x.upper()
        )
        
        # Streaming toggle
        use_streaming = st.toggle(
            "🚀 Enable streaming responses", 
            value=True,
            help="Stream responses token by token for better user experience"
        )
        
        # Note: temperature not currently used in backend API calls
        
        st.divider()
        
        # Health check
        if st.button("🏥 Health Check"):
            with st.spinner("Checking backend health..."):
                health = check_backend_health()
            
            if health['status'] == 'healthy':
                st.success("✅ Backend healthy!")
            elif health['status'] == 'initializing':
                st.warning("⏳ Backend is still initializing...")
            else:
                st.error(f"❌ Backend unhealthy: {health.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Chat management
        if st.session_state.messages:
            export_chat_history()
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.current_sources = []
                st.session_state.user_input = ""
                st.rerun()
    
    # Main layout with chat and sources
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                render_chat_message(message, message["role"] == "user")
        
        # Chat input section
        st.markdown("---")
        
        # Example questions (informational only since chat_input doesn't support programmatic input)
        st.write("**💡 Try these example questions:**")
        st.write("• What topics are covered in the document corpus?")
        st.write("• How can I search for specific information?")
        st.write("• What types of documents are available?")
        st.write("• How does the retrieval system work?")
        
        # Chat input (modern ChatGPT-style)
        if prompt := st.chat_input("Ask about your documents..."):
            timestamp = datetime.now().strftime("%H:%M")
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": timestamp
            })
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check input safety before processing
            # Note: This uses rule-based filtering - not perfect but provides basic protection
            # Production systems should use ML-based content moderation services
            safety_check = check_input_safety(prompt)
            
            if not safety_check["is_safe"]:
                # Block inappropriate content
                with st.chat_message("assistant"):
                    blocked_message = f"🚫 **Content Blocked**: {safety_check['message']}"
                    st.markdown(blocked_message)
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": blocked_message,
                    "timestamp": timestamp,
                    "safety_blocked": True
                })
                
                st.rerun()
                return  # Don't process further
            
        # Show warning for sensitive topics
        if safety_check.get("issue_type") == "sensitive_topic":
            with st.chat_message("assistant"):
                warning_msg = f"⚠️ **Sensitive Topic Detected**: {safety_check['warning']}"
                st.warning(warning_msg)
        
        elif safety_check.get("issue_type") == "off_topic":
            with st.chat_message("assistant"):
                warning_msg = f"💡 **Off-Topic Notice**: {safety_check['warning']}"
                st.info(warning_msg)
        
        # Generate and stream assistant response
        with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                if use_streaming:
                    # Use real SSE streaming
                    with st.spinner("🔍 Searching knowledge base..."):
                        time.sleep(0.3)  # Brief pause for visual feedback
                    
                    with st.spinner("🤔 Generating response..."):
                        # Get payload for streaming
                        payload = query_rag_backend(
                            question=prompt,
                            k=num_docs,
                            file_type=file_type_filter,
                            stream=True
                        )
                        
                        # Small delay to show the spinner
                        time.sleep(0.2)
                    
                    # Show initial loading state
                    message_placeholder.markdown("Thinking...")
                    
                    # Stream the response
                    result = stream_rag_response(payload, message_placeholder)
                    
                    if result.get('error'):
                        full_response = f"❌ Streaming Error: {result['error']}"
                        message_placeholder.markdown(full_response)
                        sources = []
                        retrieval_metadata = {}
                    else:
                        # Extract data from streaming result
                        response_text = result.get('answer', '')
                        sources = result.get('sources', [])
                        retrieval_metadata = result.get('retrieval_metadata', {})
                        st.session_state.current_sources = sources
                        st.session_state.current_metadata = retrieval_metadata
                        
                        # Add safety and confidence indicators
                        confidence_indicator = format_confidence_indicator(retrieval_metadata)
                        output_safety = check_output_safety(response_text)
                        safety_warnings = format_safety_warnings(output_safety)
                        response_with_disclaimer = add_contextual_disclaimer(prompt, response_text)
                        
                        # Combine all elements for final response
                        final_response = response_with_disclaimer + safety_warnings + confidence_indicator
                        full_response = final_response
                        
                        # Update with final formatted response
                        message_placeholder.markdown(full_response)
                
                else:
                    # Use traditional non-streaming approach
                    with st.spinner("🔍 Searching knowledge base..."):
                        time.sleep(0.5)  # Brief pause for visual feedback
                        
                    with st.spinner("📚 Retrieving relevant documents..."):
                        time.sleep(0.5)
                        
                    with st.spinner("🤔 Generating response..."):
                        # Query the RAG backend
                        result = query_rag_backend(
                            question=prompt,
                            k=num_docs,
                            file_type=file_type_filter,
                            stream=False
                        )
                    
                    # Check for errors
                    if result.get('error') is not None or result.get('answer') is None:
                        error_msg = result.get('error', 'Unknown error - answer is None')
                        full_response = f"❌ Error: {error_msg}"
                        message_placeholder.markdown(full_response)
                        sources = []
                        retrieval_metadata = {}
                    else:
                        response_text = result['answer']
                        sources = result.get('sources', [])
                        retrieval_metadata = result.get('retrieval_metadata', {})
                        st.session_state.current_sources = sources
                        st.session_state.current_metadata = retrieval_metadata
                        
                        # Add confidence indicator
                        confidence_indicator = format_confidence_indicator(retrieval_metadata)
                        
                        # Check output safety and bias
                        output_safety = check_output_safety(response_text)
                        
                        # Add safety warnings for output issues
                        safety_warnings = format_safety_warnings(output_safety)
                        
                        # Add contextual disclaimer based on question content
                        response_with_disclaimer = add_contextual_disclaimer(prompt, response_text)
                        
                        # Combine all elements
                        final_response = response_with_disclaimer + safety_warnings + confidence_indicator
                        
                        # Simulate streaming by displaying text progressively
                        words = final_response.split()
                        for i, word in enumerate(words):
                            full_response += word + " "
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.05)  # Adjust speed as needed
                        
                        # Final message without cursor
                        message_placeholder.markdown(full_response)
                
                # Add assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "timestamp": timestamp,
                    "sources": sources
                })
            
            st.rerun()
    
    with col2:
        st.subheader("📚 Sources & Analytics")
        
        # Sources panel with metadata
        render_sources_panel(st.session_state.current_sources, st.session_state.current_metadata)
        
    # Footer disclaimer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 20px;'>
        ⚠️ This AI assistant provides information only. 
        Always verify important details with original sources when making decisions.
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
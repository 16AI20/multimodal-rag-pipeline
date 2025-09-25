# Multimodal RAG Pipeline

## Project Overview

This project implements a **locally-deployable** Retrieval-Augmented Generation (RAG) system designed for processing multimodal content including HTML documents, PDFs, audio files, images, and structured data to provide accurate, contextual responses based on any document corpus.

### 🚀 **Key Features**
- **🔧 Modular Configuration**: Hydra-based configuration management with easy overrides
- **📊 Comprehensive Evaluation**: BLEU/ROUGE metrics, ChatGPT comparisons, automated assessment
- **🎯 Hyperparameter Tuning**: Optuna-based optimization for retrieval and generation parameters
- **📈 Observability**: Langfuse integration for monitoring, tracking, and performance analysis
- **⚡ High Performance**: Streaming responses, cross-encoder reranking, optimized embeddings
- **🏠 Local First**: Privacy-preserving, no external API dependencies, runs on local hardware
- **🔄 Real-time Streaming**: Server-Sent Events (SSE) for responsive user experience
- **🔀 Domain Agnostic**: Ready for any document corpus and knowledge domain

The system combines document embedding, vector search, and large language model generation to create an intelligent question-answering interface that can handle diverse content types and provide source-grounded responses.

## Quick Start

### Package Management Options

This project supports both **uv** (recommended) and **pip** for dependency management:

- **uv**: Modern, fast Python package installer (10-100x faster than pip)
  - Better dependency resolution
  - Built-in virtual environment management  
  - Lockfile support for reproducible builds
  - Cross-platform compatibility

- **pip**: Traditional Python package installer
  - Widely supported and familiar
  - Compatible with existing workflows

### Basic Setup

#### Option 1: Using uv (Recommended - Faster)
```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies with uv
uv sync --extra web

# 3. Start Ollama and download models
ollama serve
ollama pull llama3.1:8b

# 4. Add your documents to corpus/ directories
# (Replace with your own documents - see corpus structure below)

# 5. Create vector store
./run.sh --mode embed

# 6. Start services  
./run.sh --mode full
```

#### Option 2: Using pip (Traditional)
```bash
# 1. Install dependencies
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-web.txt

# 2. Start Ollama and download models
ollama serve
ollama pull llama3.1:8b

# 3. Add your documents to corpus/ directories
# (Replace with your own documents - see corpus structure below)

# 4. Create vector store
./run.sh --mode embed

# 5. Start services
./run.sh --mode full
```

### System Capabilities
- **Streaming**: Enable streaming toggle in Streamlit UI sidebar 
- **Evaluation**: Run comprehensive evaluation with `./run.sh --mode eval`
- **Hyperparameter Tuning**: Optimize performance with `./run.sh --mode tune`
- **Report Generation**: Generate structured markdown reports with `./run.sh --mode report`
- **Observability**: Enable Langfuse tracking for monitoring (optional)
- **API Testing**: Use FastAPI docs at http://localhost:8000/docs

## Folder Structure

```
rag-pipeline-system/
├── README.md                   # This file
├── run.sh                      # Main orchestration script
├── docker-compose.yml          # Container orchestration
├── requirements/               # Dependency management
│   ├── requirements.txt        # Core dependencies
│   ├── requirements-ml.txt     # ML/embedder dependencies
│   ├── requirements-web.txt    # Web services dependencies
│   ├── requirements-backend.txt # Backend API + enhanced ML dependencies
│   ├── requirements-dev.txt    # Development/testing dependencies
│   └── requirements-evaluation.txt # Evaluation dependencies
├── conf/                       # Hydra configuration management
│   ├── config.yaml            # Main system configuration
│   ├── llm/                   # LLM configurations (Ollama, HuggingFace)
│   ├── embeddings/            # Embedding model configurations
│   ├── prompts/               # Prompt templates (default, concise, detailed)
│   ├── vectordb/              # Vector database configurations
│   ├── observability/         # Langfuse observability settings
│   ├── optuna/                # Hyperparameter tuning configurations
│   └── logging.yaml           # Logging configuration
├── src/                        # Source code
│   ├── embedders/             # Document processing and embedding
│   │   ├── main_embedder.py   # Main embedding orchestrator
│   │   ├── base_embedder.py   # Base embedder class
│   │   ├── html_embedder.py   # HTML document processing
│   │   ├── html_parser.py     # HTML content extraction utility
│   │   ├── pdf_embedder.py    # PDF document processing
│   │   ├── audio_embedder.py  # Audio file processing
│   │   ├── image_embedder.py  # Image processing
│   │   ├── csv_embedder.py    # CSV data processing
│   │   └── docx_embedder.py   # Word document processing
│   ├── rag/                   # RAG pipeline components
│   │   ├── pipeline.py        # Main RAG orchestrator
│   │   ├── retriever.py       # Document retrieval logic
│   │   └── generator.py       # Answer generation
│   ├── interfaces/            # User interfaces
│   │   ├── fastapi_app.py     # REST API backend
│   │   └── streamlit_app.py   # Web UI frontend
│   ├── api/                   # API schemas and models
│   │   └── schemas.py         # Pydantic models
│   ├── utils/                 # Utility functions
│   │   ├── core.py           # Core utilities and Hydra configuration
│   │   ├── content_safety.py # Content moderation
│   │   └── ui_helpers.py     # UI utility functions
│   ├── evaluation/           # Evaluation framework
│   │   └── rag_evaluator.py  # Comprehensive RAG evaluation
│   ├── reports/              # Report generation
│   │   └── report_generator.py # Structured markdown report generator
│   ├── observability/        # Monitoring and tracking
│   │   └── langfuse_integration.py # Langfuse observability
│   ├── optuna/               # Hyperparameter optimization
│   │   ├── optuna_tuner.py   # Optuna-based tuning
│   │   └── run_tuning.py     # Tuning execution script
│   ├── tests/                # Unit and integration tests
│   │   ├── test_core.py      # Core functionality tests
│   │   ├── test_content_safety.py # Content safety tests
│   │   ├── test_embedders.py # Document processing tests
│   │   ├── test_fastapi_app.py # API endpoint tests
│   │   ├── test_retriever.py # Retrieval pipeline tests
│   │   ├── test_generator.py # Generation pipeline tests
│   │   ├── test_html_parser.py # HTML parsing tests
│   │   ├── test_database.py  # ChromaDB integration tests
│   │   ├── test_config_advanced.py # Configuration tests
│   │   └── test_rag_pipeline.py # End-to-end pipeline tests
├── corpus/                    # Document corpus (your content goes here)
│   ├── html/                 # HTML documents and web content
│   ├── pdf/                  # PDF documents and papers
│   ├── docx/                 # Word documents
│   ├── audio/                # Audio content (MP3, WAV files)
│   ├── images/               # Image assets (JPG, PNG, etc.)
│   └── csv/                  # Structured data files
├── evaluation/               # Evaluation framework
│   ├── bleu_rouge_answers.json # BLEU/ROUGE evaluation data (customize for your domain)
│   ├── chatgpt_comparison_answers.json # ChatGPT comparison data (customize)
│   └── results/             # Evaluation results and reports
├── reports/                   # Generated markdown reports
│   └── sample_questions.json  # Sample questions for report generation (customize)
├── docker/                   # Container definitions
│   ├── backend/             # FastAPI backend container
│   ├── frontend/            # Streamlit frontend container
│   └── embedder/            # Embedding pipeline container
├── vector_store/            # ChromaDB vector database (created after embedding)
├── logs/                    # Application logs
```

## Setup & Execution Instructions

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai/) (for LLM inference)
- Git

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd rag-pipeline-system
```

### 2. Install Dependencies

#### Using uv (Recommended)
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Core system with web interface
uv sync --extra web

# For development/testing
uv sync --extra dev

# For evaluation including RAGAS
uv sync --extra evaluation  

# For full ML capabilities (document processing, audio, images)
uv sync --extra ml

# For production monitoring
uv sync --extra monitoring

# Install everything at once
uv sync --all-extras

# Or install specific combinations
uv sync --extra web --extra evaluation --extra dev
```

#### Using pip (Traditional)
```bash
# Core system dependencies
pip install -r requirements/requirements.txt

# For development/testing  
pip install -r requirements/requirements-dev.txt

# For evaluation (includes RAGAS)
pip install -r requirements/requirements-evaluation.txt

# For backend API with enhanced ML features (includes requirements-web.txt)
pip install -r requirements/requirements-backend.txt

# Optional: For advanced features
pip install langfuse>=2.0.0      # Observability and monitoring
pip install optuna>=3.0.0        # Hyperparameter tuning  
pip install sentence-transformers # Cross-encoder reranking
```

### 3. Environment Variables
Create a `.env` file in the project root:
```bash
# Model Configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.1:8b"
export EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"

# Optional: Langfuse Observability
# export LANGFUSE_PUBLIC_KEY="pk_lf_..."
# export LANGFUSE_SECRET_KEY="sk_lf_..."
# export LANGFUSE_HOST="http://localhost:3000"

# Database Configuration
export CHROMA_DB_PATH="./vector_store"
export CHROMA_COLLECTION_NAME="rag_collection"

# Device Configuration (cpu, cuda, mps, auto)
export DEVICE="auto"

# Optional API Keys (if using external services)
# export HUGGINGFACE_API_TOKEN="your_hf_token_here"
```

### 4. Model Download Instructions
```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Download LLM models (in another terminal)
ollama pull llama3.1:8b
ollama pull llama3.2-vision  # For image processing (if using multimodal features)

# Embedding models are downloaded automatically on first use
```

### 5. Add Your Documents
```bash
# Place your documents in the corpus directories:
# - corpus/html/ for HTML files and web content
# - corpus/pdf/ for PDF documents
# - corpus/docx/ for Word documents  
# - corpus/audio/ for audio files (MP3, WAV)
# - corpus/images/ for images (JPG, PNG)
# - corpus/csv/ for structured data
```

### 6. Execution Commands

#### Local Execution
```bash
# Create vector store from corpus
./run.sh --mode embed

# Start FastAPI backend
./run.sh --mode backend

# Start Streamlit frontend
./run.sh --mode frontend

# Start full web stack
./run.sh --mode full

# Run evaluation
./run.sh --mode eval

# Run hyperparameter tuning
./run.sh --mode tune

# Generate structured reports from questions
./run.sh --mode report
```

#### Container Execution

**Prerequisites**: Ensure Ollama is running on the host (`ollama serve`) and vector store is created.

##### Individual Container Deployment (Recommended)
```bash
# Build containers
podman build -f docker/backend/Containerfile -t rag-backend .
podman build -f docker/frontend/Containerfile -t rag-frontend .

# Run backend (Terminal 1)
podman run --rm \
    --name rag-backend \
    -p 8000:8000 \
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    -v $(pwd)/vector_store:/app/vector_store \
    -v $(pwd)/logs:/app/logs \
    rag-backend

# Run frontend (Terminal 2)
podman run --rm \
    --name rag-frontend \
    -p 8501:8501 \
    rag-frontend
```

##### Docker Compose (Alternative)
```bash
# Using Docker
docker-compose --profile full up --build    # Full stack
docker-compose --profile api up --build     # Backend only
docker-compose --profile app up --build     # Frontend only

# Using Podman
podman-compose --profile full up --build    # Full stack
podman-compose --profile api up --build     # Backend only
podman-compose --profile app up --build     # Frontend only
```

### Access Points
- **Streamlit UI**: http://localhost:8501 (with streaming toggle)
- **FastAPI Backend**: http://localhost:8000 (SSE streaming support) 
- **API Documentation**: http://localhost:8000/docs

## Features

### 🔧 Configuration Management (Hydra)
The system uses Hydra for modular configuration management through YAML files in the `conf/` directory:

```bash
# Edit LLM configuration
vim conf/llm/ollama_default.yaml

# Edit embedding configuration 
vim conf/embeddings/bge_large.yaml

# Edit prompt templates
vim conf/prompts/default_rag.yaml

# Enable observability
vim conf/observability/langfuse_default.yaml
```

**Configuration Files:**
- `conf/llm/` - LLM configurations (Ollama, HuggingFace)
- `conf/embeddings/` - Embedding models and reranking settings
- `conf/prompts/` - Prompt templates (default, concise, detailed)
- `conf/observability/` - Langfuse monitoring settings
- `conf/optuna/` - Hyperparameter tuning parameters

### 📊 Evaluation
Run systematic evaluation of RAG performance:

```bash
# Run full evaluation suite
./run.sh --mode eval

# Includes:
# - BLEU/ROUGE metrics against reference answers
# - ChatGPT comparison analysis
# - Retrieval quality assessment
# - Response time analysis
```

### 🎯 Hyperparameter Tuning (Optuna)
Optimize RAG performance automatically:

```bash
# Enable tuning in config
# conf/optuna/optuna_default.yaml
enabled: true

# Run optimization
./run.sh --mode tune

# Optimizes:
# - Retrieval parameters (k, chunk sizes, reranking)
# - Generation parameters (temperature, max_tokens, top_p)
# - Uses evaluation metrics as objective function
```

### 📈 Observability (Langfuse)
Monitor RAG pipeline performance:

```bash
# Set up Langfuse (cloud or self-hosted)
export LANGFUSE_PUBLIC_KEY="pk_lf_..."
export LANGFUSE_SECRET_KEY="sk_lf_..."

# Enable in config
# conf/observability/langfuse_default.yaml
enabled: true

# Tracks:
# - Query performance and latency
# - Retrieval quality metrics
# - Token usage and costs
# - User sessions and feedback
```

## Key Components & Architecture

This system is designed for **local deployment** with **minimal compute requirements**, making it accessible for various use cases while maintaining production-level capabilities.

### Embedding Model: BAAI/bge-large-en-v1.5
- **Rationale**: State-of-the-art multilingual embedding model with strong performance on English text
- **Technical Advantages**: High-quality semantic representations, good domain adaptability, runs efficiently on CPU/GPU
- **Local Deployment Benefits**: No external API costs, complete data privacy, consistent performance
- **Configuration**: 1024-dimensional embeddings with L2 normalization for cosine similarity optimization

### Vector Database: ChromaDB
- **Rationale**: Modern, lightweight vector database with minimal infrastructure requirements
- **Technical Advantages**: Python-native integration, built-in similarity search, efficient metadata filtering
- **Configuration**: Cosine similarity matching, persistent disk storage, separate collections for different content types

### LLM: Ollama/Llama3.1:8b & Flexible Configuration
- **Rationale**: Local deployment for privacy and cost constraints
- **Resource Optimization**: `llama3.1:8b` provides optimal balance of performance and hardware requirements
- **Flexibility**: HuggingFace models supported via Hydra configuration
- **Privacy Benefits**: Complete data locality ensures compliance with data protection requirements

### UI Framework: Streamlit + FastAPI
- **Streamlit Frontend**: Rapid prototyping and intuitive user interface
- **FastAPI Backend**: High-performance API serving with automatic OpenAPI documentation
- **Microservices Architecture**: Separate frontend/backend enables flexible deployment models
- **Advanced Features**: Server-Sent Events (SSE) streaming for real-time response experience

## API Features

### FastAPI Backend Endpoints
- **GET /health** - Health check with component status
- **GET /collection** - Vector database collection information  
- **POST /query** - Full RAG query with streaming support
- **POST /retrieve** - Document retrieval only
- **GET /docs** - Interactive Swagger documentation

### Streaming Support
The system supports **Server-Sent Events (SSE)** for real-time streaming responses:

#### Streaming Query
```bash
# Enable streaming in request
POST /query
{
  "question": "What topics are covered?",
  "stream": true,
  "k": 5
}
```

#### Response Flow
1. **Sources First**: Retrieved documents sent immediately
2. **Token Streaming**: LLM response streamed token-by-token
3. **Completion**: Final response with metadata

## Customizing for Your Domain

### 1. Document Corpus
Replace the `corpus/` directory contents with your documents:
- `corpus/html/` - HTML files, web content, documentation
- `corpus/pdf/` - PDF documents, papers, manuals
- `corpus/docx/` - Word documents, reports
- `corpus/audio/` - Audio files, recordings, podcasts
- `corpus/images/` - Images, diagrams, charts
- `corpus/csv/` - Structured data, tables

### 2. Evaluation Setup
Update evaluation files for your domain:
- `evaluation/bleu_rouge_answers.json` - Questions and reference answers for your topic
- `evaluation/chatgpt_comparison_answers.json` - ChatGPT comparison questions
- `reports/sample_questions.json` - Sample questions for report generation

### 3. Configuration Customization
- `conf/prompts/` - Update assistant descriptions for your domain
- `src/utils/content_safety.py` - Add domain-specific relevance terms
- `conf/config.yaml` - Adjust chunk sizes for your content types

### 4. Interface Customization
- `src/interfaces/streamlit_app.py` - Update titles, descriptions, example questions
- `src/interfaces/fastapi_app.py` - Customize API documentation

## Evaluation Framework

### 📊 Multi-Dimensional Assessment
The system includes a comprehensive evaluation framework accessible via `./run.sh --mode eval`:

### 1. **ChromaDB Retrieval Metrics**
- **Success Rate** (0-100%): Percentage of queries returning meaningful results
- **Retrieval Confidence** (0-1): Average similarity score between queries and retrieved documents
- **Source Diversity** (1-3): Average number of different document types retrieved

### 2. **BLEU/ROUGE vs Reference Answers**
- **BLEU Score** (0-1): Measures exact n-gram overlap
- **ROUGE-1/2/L F1** (0-1): Unigram, bigram, and longest common subsequence matching
- **Semantic Similarity** (0-1): BERT-based embedding comparison

### 3. **RAG vs ChatGPT Comparison**
- **Response Similarity** (0-1): Semantic similarity between RAG and ChatGPT answers
- **Consistency Analysis**: Identifies systematic differences in response patterns

## Report Generation

Generate structured markdown reports from question files:

```bash
# Direct command line usage
./run.sh --mode report
```

**Input Format**: JSON file with questions array:
```json
{
  "questions": [
    "What topics are covered in the documents?",
    "How can I search for specific information?",
    "What types of documents are supported?"
  ]
}
```

**Output**: Professional markdown report with Q&A format, timestamps, and source citations.

## Performance Optimizations

### Hyperparameter Optimization Results
Baseline optimized parameters achieving good performance:

```yaml
# Retrieval Settings
k: 8                        # Retrieve 8 documents for balanced context
chunk_size_pdf: 1100        # Moderate PDF chunks for dense content  
chunk_size_html: 1700       # Larger HTML chunks for web content structure
chunk_overlap: 50           # Minimal overlap for clean boundaries
similarity_threshold: 0.4   # Moderate selectivity for relevance
reranking_enabled: false    # Test with your domain for effectiveness

# Generation Settings  
temperature: 0.2            # Low creativity for consistent, factual responses
max_tokens: 832             # Balanced response length
```

### Performance Features
- **Cross-encoder Reranking**: Optional improved relevance ranking
- **Streaming Responses**: Real-time token generation
- **Caching**: Efficient vector similarity caching
- **Multimodal Processing**: Optimized for different document types

## Security & Privacy

### Content Safety Features
- **Multi-layered Content Filtering**: Rule-based analysis with context-aware safety checks
- **Response Validation**: Output filtering before delivery to users
- **Source Attribution**: All responses include verifiable citations

### Privacy Design
- **Local Processing**: All data processing occurs on-premises
- **No External Dependencies**: Complete offline capability
- **Audit Trails**: Comprehensive logging for transparency
- **Data Locality**: No data leaves your infrastructure

## Docker Deployment

### Environment Variables for Containers
- `TOKENIZERS_PARALLELISM=false` - Prevents HuggingFace tokenizer deadlocks
- `PYTORCH_ENABLE_MPS_FALLBACK=1` - Apple Silicon MPS compatibility
- `PYTHONPATH=/app` - Ensures proper Python module loading

### Container Profiles
- `api` - Backend API only
- `app` - Frontend only  
- `full` - Complete stack

## License

This project is licensed under the MIT License - see the LICENSE file for details.

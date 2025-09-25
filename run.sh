#!/bin/bash

# RAG Pipeline Orchestrator
# This script manages the complete RAG system including:
# - Vector store embedding pipeline
# - Interactive CLI interface  
# - FastAPI backend server
# - Streamlit frontend interface

set -e  # Exit on any error

# Default mode
MODE="embed"

# Check if uv is available and working
check_uv_available() {
    if command -v uv &> /dev/null; then
        if uv --version &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Run Python command with uv if available, otherwise use system python
run_python() {
    if check_uv_available; then
        uv run python "$@"
    else
        python "$@"
    fi
}

# Install dependencies with uv if available, otherwise fall back to pip
install_deps_if_needed() {
    local extra="$1"
    local pip_requirements="$2"
    
    if check_uv_available; then
        echo "📦 Using uv for dependency management..."
        if [ ! -z "$extra" ]; then
            uv sync --extra "$extra" --quiet
        else
            uv sync --quiet
        fi
    else
        echo "📦 Using pip for dependency management..."
        if [ ! -z "$pip_requirements" ] && [ -f "$pip_requirements" ]; then
            pip install -r "$pip_requirements" --quiet
        fi
    fi
}

# Check Python installation
check_python_deps() {
    local module="$1"
    if check_uv_available; then
        uv run python -c "import $module" 2>/dev/null
    else
        python -c "import $module" 2>/dev/null
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "RAG Pipeline Orchestrator"
            echo ""
            echo "Usage: $0 [--mode MODE]"
            echo ""
            echo "Modes:"
            echo "  embed     - Create/update vector store from corpus (default)"
            echo "  backend   - Start FastAPI backend server"
            echo "  frontend  - Start Streamlit web interface"
            echo "  full      - Start both backend and frontend"
            echo "  eval      - Run RAG system evaluation"
            echo "  tune      - Run hyperparameter tuning with Optuna"
            echo "  report    - Generate structured markdown reports"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run embedding pipeline"
            echo "  $0 --mode backend     # Start FastAPI backend"
            echo "  $0 --mode frontend    # Start Streamlit web interface"
            echo "  $0 --mode full        # Start both backend and frontend"
            echo "  $0 --mode eval        # Run evaluation (requires templates)"
            echo "  $0 --mode tune        # Run hyperparameter tuning"
            echo "  $0 --mode report      # Generate reports (interactive mode)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "RAG Pipeline Orchestrator"
echo "Mode: $MODE"
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/embedders/main_embedder.py" ]; then
    echo "❌ Error: Please run this script from the project root directory."
    exit 1
fi

# Check if corpus directory exists
if [ ! -d "corpus" ]; then
    echo "❌ Error: corpus directory not found. Please ensure corpus data is available."
    exit 1
fi

echo "✅ Environment checks passed"

# Function to start Ollama if needed
start_ollama() {
    echo "🔍 Checking Ollama setup..."
    
    if ! command -v ollama &> /dev/null; then
        echo "⚠️  Ollama not found. Some features may be limited."
        echo "💡 Install Ollama: https://ollama.ai/"
        return 1
    fi
    
    # Check if Ollama server is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "🚀 Starting Ollama server..."
        
        # Start Ollama in background
        ollama serve &
        OLLAMA_PID=$!
        
        # Wait for Ollama to start
        echo "⏳ Waiting for Ollama to start..."
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "✅ Ollama server started successfully"
                break
            fi
            sleep 1
            if [ $i -eq 30 ]; then
                echo "❌ Failed to start Ollama server"
                return 1
            fi
        done
    else
        echo "✅ Ollama server already running"
    fi
    
    return 0
}

# Function to verify Ollama is accessible (models will be checked by the pipeline)
check_ollama_ready() {
    if ! command -v ollama &> /dev/null; then
        echo "⚠️  Ollama not found. Pipeline will handle model requirements."
        return 1
    fi
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "⚠️  Ollama server not running. Pipeline will handle connection."
        return 1
    fi
    
    echo "✅ Ollama server is accessible"
    return 0
}

# Ensure logs directory exists
mkdir -p logs
echo "📁 Created logs directory for logging output"
echo ""

# Show current corpus contents
echo "📁 Corpus Directory Contents:"
echo "--------------------------------------------"
for dir in corpus/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        case "$dirname" in
            "audio")
                count=$(find "$dir" -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.flac" -o -name "*.m4a" -o -name "*.ogg" \) | wc -l)
                ;;
            "csv")
                count=$(find "$dir" -type f -name "*.csv" | wc -l)
                ;;
            "docx")
                count=$(find "$dir" -type f -name "*.docx" | wc -l)
                ;;
            "html")
                count=$(find "$dir" -type f -name "*.html" | wc -l)
                ;;
            "images")
                count=$(find "$dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" -o -name "*.bmp" -o -name "*.webp" \) | wc -l)
                ;;
            "pdf")
                count=$(find "$dir" -type f -name "*.pdf" | wc -l)
                ;;
            *)
                count=$(find "$dir" -type f -not -name ".*" | wc -l)
                ;;
        esac
        echo "  $dirname: $count files"
    fi
done
echo ""

# Mode-specific pre-checks
case $MODE in
    "embed")
        # Check if vector store already exists (only for embed mode)
        if [ -d "vector_store" ]; then
            echo "⚠️  Existing vector store found at ./vector_store"
            echo ""
            echo "Current collection info:"
            run_python -m src.embedders.main_embedder --info-only
            echo ""
            
            read -p "Do you want to proceed? This will add to the existing collection. (y/N): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "❌ Aborted by user"
                exit 1
            fi
            echo ""
        fi
        ;;
esac

# Set common environment variables for all modes that use the pipeline
set_pipeline_env() {
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
}

# Function to handle cleanup for frontend mode
cleanup_frontend_mode() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $API_PID 2>/dev/null || true
    exit 0
}

# Function to handle cleanup for full mode
cleanup_full_mode() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $API_PID 2>/dev/null || true
    exit 0
}

# Mode-specific execution
case $MODE in
    "embed")
        echo "🚀 Starting vector store embedding pipeline..."
        echo "This may take several minutes depending on corpus size."
        echo ""
        
        # Set environment variables
        set_pipeline_env
        
        # Start Ollama for image processing
        if start_ollama; then
            check_ollama_ready
        fi
        
        # Run the main embedder
        install_deps_if_needed "ml" "requirements/requirements-ml.txt"
        run_python -m src.embedders.main_embedder
        
        echo ""
        echo "✅ Vector store embedding complete!"
        echo ""
        
        # Show final collection info
        echo "📊 Final Collection Information:"
        echo "--------------------------------------------"
        run_python -m src.embedders.main_embedder --info-only
        
        echo ""
        echo "🎉 Your RAG vector store is ready!"
        echo "📍 Location: ./vector_store"
        echo "📝 Collection: rag_collection"
        ;;
        
        
    "backend")
        echo "🚀 Starting FastAPI backend server..."
        
        # Check if uvicorn is available
        if ! check_python_deps "uvicorn"; then
            echo "❌ uvicorn not found. Installing dependencies..."
            install_deps_if_needed "web" "requirements/requirements-web.txt"
            exit 1
        fi
        
        # Set environment variables
        set_pipeline_env
        
        # Start Ollama if possible
        start_ollama && check_ollama_ready
        
        # Start the FastAPI server
        echo ""
        echo "🌐 Backend server will be available at: http://localhost:8000"
        echo "📖 Interactive API docs at: http://localhost:8000/docs"
        echo "🔧 Starting FastAPI backend (pipeline will validate configuration)..."
        echo ""
        uvicorn src.interfaces.fastapi_app:app --reload --host 0.0.0.0 --port 8000
        ;;
        
    "frontend")
        echo "🚀 Starting Streamlit frontend with backend..."
        
        # Check dependencies
        if ! check_python_deps "uvicorn" || ! check_python_deps "streamlit"; then
            echo "❌ Missing dependencies. Installing..."
            install_deps_if_needed "web" "requirements/requirements-web.txt"
        fi
        
        # Set environment variables
        set_pipeline_env
        
        # Start Ollama if possible
        start_ollama && check_ollama_ready
        
        # Set up signal handling for cleanup
        trap cleanup_frontend_mode SIGINT SIGTERM
        
        echo ""
        echo "🌐 FastAPI backend: http://localhost:8000"
        echo "🎨 Streamlit frontend: http://localhost:8501"
        echo ""
        echo "🔧 Starting both services (pipeline will validate configuration)..."
        echo "💡 Press Ctrl+C to stop all services"
        echo ""
        
        # Start FastAPI backend in background
        echo "Starting FastAPI backend..."
        uvicorn src.interfaces.fastapi_app:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        
        # Wait for backend to start
        sleep 3
        
        # Start Streamlit frontend in foreground
        echo "Starting Streamlit frontend..."
        streamlit run src/interfaces/streamlit_app.py --server.port 8501 --server.headless true
        ;;
        
    "full")
        echo "🚀 Starting full web stack (Backend + Frontend)..."
        
        # Check dependencies
        if ! python -c "import uvicorn, streamlit" 2>/dev/null; then
            echo "❌ Missing dependencies. Install with: pip install uvicorn streamlit"
            exit 1
        fi
        
        # Set environment variables
        set_pipeline_env
        
        # Start Ollama if possible
        start_ollama && check_ollama_ready
        
        # Set up signal handling for cleanup
        trap cleanup_full_mode SIGINT SIGTERM
        
        echo ""
        echo "🌐 FastAPI backend: http://localhost:8000"
        echo "📖 Backend API docs: http://localhost:8000/docs"
        echo "🎨 Streamlit frontend: http://localhost:8501"
        echo ""
        echo "🔧 Starting both services (pipeline will validate configuration)..."
        echo "💡 Press Ctrl+C to stop all services"
        echo ""
        
        # Start FastAPI backend in background
        echo "Starting FastAPI backend..."
        uvicorn src.interfaces.fastapi_app:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        
        # Wait for backend to start
        sleep 3
        
        # Start Streamlit frontend in foreground
        echo "Starting Streamlit frontend..."
        streamlit run src/interfaces/streamlit_app.py --server.port 8501 --server.headless true
        
        # Cleanup: this will be called by the trap
        ;;
        
    "eval")
        echo "🧪 Running RAG System Evaluation..."
        
        # Check evaluation dependencies
        if ! python -c "import nltk, rouge_score, sentence_transformers" 2>/dev/null; then
            echo "❌ Missing evaluation dependencies. Install with:"
            echo "   pip install -r requirements/requirements-evaluation.txt"
            exit 1
        fi
        
        # Set environment variables
        set_pipeline_env
        
        # Start Ollama for LLM evaluation
        start_ollama && check_ollama_ready
        
        # Check if evaluation templates exist
        BLEU_ROUGE_FILE=""
        CHATGPT_FILE=""
        
        if [ -f "evaluation/bleu_rouge_answers.json" ]; then
            BLEU_ROUGE_FILE="--bleu-rouge evaluation/bleu_rouge_answers.json"
            echo "✅ Found BLEU/ROUGE answers file"
        else
            echo "⚠️ BLEU/ROUGE template not found (evaluation/bleu_rouge_answers.json)"
            echo "   Fill evaluation/bleu_rouge_template.json and save as bleu_rouge_answers.json"
        fi
        
        if [ -f "evaluation/chatgpt_comparison_answers.json" ]; then
            CHATGPT_FILE="--chatgpt evaluation/chatgpt_comparison_answers.json"
            echo "✅ Found ChatGPT comparison answers file"
        else
            echo "⚠️ ChatGPT comparison template not found (evaluation/chatgpt_comparison_answers.json)"
            echo "   Fill evaluation/chatgpt_comparison_template.json and save as chatgpt_comparison_answers.json"
        fi
        
        # Create output directory
        mkdir -p evaluation/results
        
        # Run evaluation without timestamp
        OUTPUT_JSON="evaluation/results/evaluation_results.json"
        OUTPUT_MD="evaluation/results/evaluation_report.md"
        
        echo "🔧 Running evaluation..."
        python src/evaluation/rag_evaluator.py \
            $BLEU_ROUGE_FILE \
            $CHATGPT_FILE \
            --output "$OUTPUT_JSON" \
            --markdown "$OUTPUT_MD"
        
        if [ $? -eq 0 ]; then
            echo "✅ Evaluation complete!"
            echo "📊 JSON Results: $OUTPUT_JSON"
            echo "📝 Markdown Report: $OUTPUT_MD"
            
            # Show results location
            echo ""
            echo "📁 Results saved in evaluation/results/"
            ls -la evaluation/results/ | tail -5
        else
            echo "❌ Evaluation failed. Check error messages above."
            exit 1
        fi
        ;;
        
    "tune")
        echo "🔧 Running Hyperparameter Tuning with Optuna..."
        
        # Check tuning dependencies
        if ! python -c "import optuna" 2>/dev/null; then
            echo "❌ Optuna not found. Install with: pip install optuna"
            exit 1
        fi
        
        # Set environment variables
        set_pipeline_env
        
        # Check if evaluation files exist
        BLEU_ROUGE_FILE=""
        CHATGPT_FILE=""
        
        if [ -f "evaluation/bleu_rouge_answers.json" ]; then
            BLEU_ROUGE_FILE="--bleu-rouge evaluation/bleu_rouge_answers.json"
            echo "✅ Found BLEU/ROUGE answers file"
        else
            echo "❌ BLEU/ROUGE answers file missing (evaluation/bleu_rouge_answers.json)"
            echo "   Run evaluation setup first or create the file"
            exit 1
        fi
        
        if [ -f "evaluation/chatgpt_comparison_answers.json" ]; then
            CHATGPT_FILE="--chatgpt evaluation/chatgpt_comparison_answers.json"
            echo "✅ Found ChatGPT comparison answers file"
        else
            echo "⚠️  ChatGPT comparison file not found (optional)"
        fi
        
        # Create output directory
        mkdir -p hyperparameter_results
        
        # Run hyperparameter tuning with timestamp
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_DB="hyperparameter_results/study_${TIMESTAMP}.db"
        
        echo "🔧 Starting hyperparameter optimization..."
        echo "💡 This may take a while depending on n_trials configuration"
        echo "📊 Study database: $OUTPUT_DB"
        echo ""
        
        python src/optuna/run_tuning.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Hyperparameter tuning complete!"
            echo "📊 Check hyperparameter_results/ for detailed results"
            echo "🔧 Optimized config saved (if enabled in configuration)"
            echo "💡 Start Optuna dashboard: optuna-dashboard sqlite:///hyperparameter_studies.db"
        else
            echo "❌ Hyperparameter tuning failed. Check error messages above."
            exit 1
        fi
        ;;
        
    "report")
        echo "📝 Starting Report Generation Pipeline..."
        
        # Set environment variables
        set_pipeline_env
        
        # Start Ollama for report generation
        start_ollama && check_ollama_ready
        
        # Create reports directory if it doesn't exist
        mkdir -p reports
        
        echo ""
        echo "📋 Report Generation from Questions File"
        echo "--------------------------------------------"
        
        # Check if sample questions file exists
        if [ -f "reports/sample_questions.json" ]; then
            echo "✅ Found sample questions file: reports/sample_questions.json"
            echo ""
            echo "📋 Sample questions preview:"
            python -c "import json; data=json.load(open('reports/sample_questions.json')); [print(f'  • {q}') for q in data['questions'][:3]]; print(f'  ... and {len(data[\"questions\"])-3} more questions')" 2>/dev/null || echo "  (Error reading sample file)"
            echo ""
            
            read -p "Use reports/sample_questions.json? (Y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                echo ""
                read -p "Enter path to your questions JSON file: " questions_file
            else
                questions_file="reports/sample_questions.json"
            fi
        else
            echo "⚠️  reports/sample_questions.json not found"
            echo ""
            read -p "Enter path to your questions JSON file: " questions_file
        fi
        
        # Validate questions file exists
        if [ ! -f "$questions_file" ]; then
            echo "❌ Questions file not found: $questions_file"
            echo ""
            echo "💡 Create a JSON file with this format:"
            echo '   {"questions": ["Question 1?", "Question 2?"]}'
            echo ""
            echo "📄 Example: reports/sample_questions.json"
            exit 1
        fi
        
        # Generate output filename
        timestamp=$(date +%Y%m%d_%H%M%S)
        output_file="reports/report_${timestamp}.md"
        
        echo ""
        echo "🔧 Generating report from: $questions_file"
        echo "📄 Output file: $output_file"
        echo ""
        
        # Run report generation
        python -m src.reports.report_generator --questions "$questions_file" --output "$output_file" --verbose
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Report generation complete!"
            echo "📄 Report saved: $output_file"
            echo ""
            echo "💡 You can also run directly:"
            echo "   python -m src.reports.report_generator --questions $questions_file --output custom_report.md"
        else
            echo "❌ Report generation failed"
            exit 1
        fi
        ;;
        
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Use --help for available modes"
        exit 1
        ;;
esac
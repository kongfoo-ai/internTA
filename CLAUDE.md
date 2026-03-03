# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InternTA is a multi-agent AI teaching assistant system for synthetic biology education. It uses a three-agent architecture:
1. **Dataset Agent** - Generates high-quality training data with explicit reasoning paths
2. **Training Agent** - Fine-tunes models using knowledge distillation with limited data
3. **RAG Agent** - Enhances responses by retrieving external knowledge

The system is designed for local deployment to address data privacy concerns in educational settings.

## Key Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set API token (optional)
echo "API_TOKEN=your-secret-token" > .env
```

### Running the Application
```bash
# Start the Streamlit web interface (default port 8080)
sh run.sh

# Alternative: Direct Streamlit command
CUDA_VISIBLE_DEVICES=0 streamlit run app.py --server.address=0.0.0.0 --server.port 8080 --server.fileWatcherType none -- --show-local-option --local
```

### Training Pipeline
```bash
# Generate training data
cd data
python generate_data.py

# Train the model (basic)
sh train.sh

# Train with advanced agent-based training
sh traino.sh

# Merge fine-tuned adapter into base model
sh merge.sh $NUM_EPOCH
```

### Testing and Evaluation
```bash
# Run model evaluation tests
pytest ./test/test_model_evaluation.py

# Test API authentication
python test/test_auth.py
```

## Architecture

### Core Components
- **app.py** - Streamlit web interface with local/remote mode switching
- **api.py** - FastAPI server with OpenAI-compatible endpoints
- **data/** - Dataset generation and processing scripts
- **train/** - Model training scripts and configuration
- **test/** - Evaluation and testing scripts

### Model Configuration
- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Fine-tuning: QLoRA (4-bit quantization) with PEFT
- Training approach: Knowledge distillation from larger models

### API Structure
- Endpoint: `/v1/chat/completions` (OpenAI-compatible)
- Authentication: Bearer token via `API_TOKEN` environment variable
- Local model loading: 4-bit quantized with LoRA adapters

## Development Notes

### Environment Requirements
- NVIDIA GPU with 8GB+ VRAM recommended for local model operation
- Python dependencies include PyTorch, Transformers, PEFT, TRL, and Streamlit
- CUDA 11.8+ required for GPU acceleration

### Data Flow
1. Raw data from Excel files → `data/generate_data.py`
2. Generated training/validation JSON files → `data/training.json`, `data/validation.json`
3. Model training → `train/train_agent.py` or `train/sft_internTA2.py`
4. Model serving → `api.py` (FastAPI) or `app.py` (Streamlit)

### Testing Strategy
- Model evaluation uses ROUGE similarity scores
- Authentication tests verify API token functionality
- Integration tests validate end-to-end pipeline

### Configuration Files
- `config/internlm2_1_8b_qlora_alpaca_e3_copy.py` - Training configuration
- `.env` - API token and environment variables
- `requirements.txt` - Python dependencies
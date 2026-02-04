# Multimodal Jailbreak Detector
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-red.svg)](https://pytorch.org/)

A multimodal (text + image) LLM jailbreak prompt detector based on **K-NB Rank + PyTorch IsolationForest**, supporting LLaVA-1.6 and Qwen-VL models. It efficiently identifies malicious jailbreak prompts and outputs anomaly scores without fine-tuning the base model.

## 🌟 Key Features
- **Multimodal Support**: Detect jailbreak prompts with text-image mixed inputs (compatible with LLaVA/Qwen-VL dual models)
- **Efficient Anomaly Detection**: K-NB Rank feature extraction + PyTorch-based IsolationForest (GPU-accelerated training/inference)
- **Comprehensive Evaluation Metrics**: AUROC, AUPRC, FPR, Precision, and other core metrics
- **Multi-Dataset Compatibility**: Built-in loaders for MM-SafetyBench, SD-AdvBench, MM-Vet v2, VQA datasets
- **Low Intrusion**: No fine-tuning required for large models (uses hidden layer activations for detection)

## 📋 Environment Requirements
### Basic Environment
- Python 3.8+ (3.9/3.10 recommended for PyTorch compatibility)
- CUDA 11.7+ (recommended for GPU acceleration; CPU is supported but slower)
- GPU Memory: ≥16GB for LLaVA-1.6-Vicuna-7B, ≥12GB for Qwen-VL-Chat

### Dependency Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/multimodal-jailbreak-detector.git
cd multimodal-jailbreak-detector

# Install core dependencies
pip install -r requirements.txt

# (Optional) For CPU-only PyTorch
# pip install torch --index-url https://download.pytorch.org/whl/cpu
🚀 Quick Start
1. Dataset Preparation
Place datasets in the specified paths (modify paths load_datasets.pypy`):
Dataset	Path Example	Purpose
VQA	./datasets/vqa/test2015	Benign samples (training)
MM-Vet v2	./datasets/mm-vet-v2	Benign samples (testing)
SD-AdvBench	./datasets/sd_advbench	Malicious samples (training)
MM-SafetyBench	./datasets/MM-SafetyBench	Malicious samples (testing)
2. Model Weights Preparation
Download multimodal model weights and place them in the specified paths (modify from_pretrained paths in test scripts):
LLaVA-1.6-Vicuna-7B: /HARD-DATA/ZHT/A_model/llava-v1.6-vicuna-7b-hf
Qwen-VL-Chat: /HARD-DATA/ZHT/A_model/qwen_vl_chat
3. Run Detection (LLaVA Example)
bash
运行
# Run jailbreak detection evaluation for LLaVA
python test_AUROC_llava.py

# Run jailbreak detection evaluation for Qwen-VL
python test_AUROC_qwen.py
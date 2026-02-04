# Multimodal Jailbreak Detector



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
git clone https://github.com/XLionXL/mtk.git
cd mtk/vlm

# Install core dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Dataset Preparation

Place datasets in the specified paths (modify paths in `load_datasets.py`):

|Dataset|Path Example|Purpose|
|---|---|---|
|VQA|./datasets/vqa/test2015|Benign samples (training)|
|MM-Vet v2|./datasets/mm-vet-v2|Benign samples (testing)|
|SD-AdvBench|./datasets/sd_advbench|Malicious samples (training)|
|MM-SafetyBench|./datasets/MM-SafetyBench|Malicious samples (testing)|
### 2. Model Weights Preparation

Download multimodal model weights and place them in the specified paths (modify `from_pretrained` paths in test scripts):

- LLaVA-1.6-Vicuna-7B: `./models/llava-v1.6-vicuna-7b-hf`

- Qwen-VL-Chat: `./model/qwen_vl_chat`

### 3. Run Detection

```bash
# Run jailbreak detection evaluation for LLaVA
python test_AUROC_llava.py 

# Run jailbreak detection evaluation for Qwen-VL
python test_AUROC_qwen.py
```
> （注：文档部分内容可能由 AI 生成）
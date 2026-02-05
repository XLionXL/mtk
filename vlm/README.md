# MTK

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
|[VQA](https://visualqa.org/download.html)|./datasets/vqa/test2015|Benign samples (training)|
|[MM-Vet v2]|./datasets/mm-vet-v2|Benign samples (testing)|
|[SD-AdvBench]|./datasets/sd_advbench|Malicious samples (training)|
|[MM-SafetyBench]|./datasets/MM-SafetyBench|Malicious samples (testing)|
|[FigStep]()https://github.com/CryptoAILab/FigStep/tree/main/data/images/SafeBench|./datasets/FigStep|Malicious samples (testing)|
|[JailBreakV_28K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)|./datasets/JailBreakV_28K|Malicious samples (testing)|
|[USB-Overrefusal]|./datasets/MM-SafetyBench|Benign samples (testing)|

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

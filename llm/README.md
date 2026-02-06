# MTK

## 📋 Environment Requirements

### Basic Environment

- Python 3.8+ (3.9/3.10 recommended for PyTorch compatibility)

- CUDA 11.7+ (recommended for GPU acceleration; CPU is supported but slower)


### Dependency Installation

```bash
# Clone repository
git clone https://github.com/XLionXL/mtk.git
cd mtk/llm/llm_mtk

# Install core dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Training Datasets Preparation

Place training datasets in ./datasets/train_data:

|Dataset|Path Example|Purpose|
|---|---|---|
|[Alpaca ](https://huggingface.co/datasets/tatsu-lab/alpaca)|./datasets/train_data|Benign samples|
|[Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)|./datasets/train_data|Benign samples|
|[Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)|./datasets/train_data|Benign samples|
|[Or-Bench_80k](https://huggingface.co/datasets/bench-llm/or-bench)|./datasets/train_data|Pseudo-Malicious samples|
|[MaliciousInstruct](https://huggingface.co/datasets/walledai/MaliciousInstruct)|./datasets/train_data|Malicious samples|
|[Advbench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)|./datasets/train_data|Malicious samples|
|[PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)|./datasets/train_data|Malicious samples|



> ⚠️ Note: All datasets should be converted into a unified .txt format.

----------

### 2. Test Datasets

Place **test datasets** under:

`./datasets/{model_name}_test`

where `{model_name}` must be one of:

-   `llama2`
    
-   `llama3`
    
-   `mistral`
    
-   `vicuna`
    

#### File Naming Convention

-   **Jailbreak / malicious attack samples**:
    
    `{attack_name}_1.json`
    
-   **Benign samples**:
    
    `{benign_name}_0.json`
    

Here, the suffix `_1` indicates malicious/jailbreak data, and `_0` indicates benign data.  
This naming convention is required for correct label parsing during evaluation.

### 3. Model Weights Preparation

Download the following large language model weights and place them under:

`./model`

Required models:

-   **Llama2-7b-chat-hf**
    
-   **Llama3-8b-Instruct**
    
-   **Mistral-7b-instruct-v0.2**
    
-   **Vicuna-7b-v1.5**
    

Each model should be stored in its own subdirectory following the Hugging Face standard structure.


### 4. Run Detection

```bash
# Run jailbreak detection evaluation for llama2
python mtk_llama2.py

# Run jailbreak detection evaluation for llama3
python mtk_llama3.py

# Run jailbreak detection evaluation for mitstral
python mtk_mistral.py

# Run jailbreak detection evaluation for vicuna
python mtk_vicuna.py
```

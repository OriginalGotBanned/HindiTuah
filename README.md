# English to Hindi Translation with AI4Bharat

This guide will help you set up and run an English to Hindi translation model using AI4Bharat's IndicTrans2 with LoRA (Low-Rank Adaptation). No prior ML experience required!

## 🚀 Quick Start Guide

### Step 1: System Requirements

#### Minimum Requirements
- Windows 10/11 or Linux
- 16GB RAM (8GB minimum)
- 10GB free disk space
- NVIDIA GPU with 8GB+ VRAM (highly recommended)
- Python 3.8 or later

#### Recommended
- NVIDIA RTX 3060 or better
- 32GB RAM
- SSD storage
- CUDA 11.8 or later

### Step 2: Install Dependencies

1. **Install Python**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Verify GPU and CUDA**
   - Press `Windows + R`, type `cmd`, and press Enter
   - Run: `nvidia-smi`
   - You should see your GPU information. If not:
     1. Update your NVIDIA drivers from [here](https://www.nvidia.com/download/index.aspx)
     2. Install CUDA Toolkit from [here](https://developer.nvidia.com/cuda-downloads)
     3. Restart your computer

3. **Install Required Packages**
   Open Command Prompt and run:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers==4.52.4 peft==0.15.2 datasets bitsandbytes tqdm scikit-learn numpy protobuf
   ```

## 🛠 How to Use the Scripts

### 1. Check Your Setup
```bash
python check_gpu.py
```
This verifies your GPU and CUDA installation.

### 2. Prepare Your Data

#### Option 1: Using Your Own Data
Place your English-Hindi translation pairs in a CSV file with columns: `prompt` (English) and `response` (Hindi).

#### Option 2: Using BPCC Dataset from Hugging Face
1. **Install required packages** (if not already installed):
   ```bash
   pip install datasets
   ```

2. **Run the data loader script**:
   ```bash
   python load_bpcc.py --output_dir data/processed --language_pair en-hi
   ```
   
   **Parameters:**
   - `--output_dir`: Directory to save processed data (default: 'data/processed')
   - `--language_pair`: Language pair code (e.g., 'en-hi' for English-Hindi)
   - `--split`: Dataset split to download ('train', 'validation', 'test')
   - `--max_samples`: Maximum number of samples to process (optional)
   - `--cache_dir`: Cache directory for Hugging Face datasets (optional)

3. **Verify the processed data**:
   ```bash
   ls -l data/processed/
   ```
   You should see files like `train.csv`, `validation.csv`, and `test.csv`.

### 3. Inspect Your Data
```bash
python inspect_dataset.py --input your_data.csv
```
This shows sample translations and basic statistics.

### 4. (Optional) Generate More Training Data
```bash
python generate_synthetic_data.py --input your_data.csv --output synthetic_data.csv --num_examples 1000
```

### 5. Train the Model
```bash
python train_lora.py --data your_data.csv --output_dir my_translation_model
```

### 6. Test the Model
```bash
python test_model.py --model_dir my_translation_model
```

## 🧩 Script Descriptions

### Core Scripts

#### `train_lora.py`
Trains the translation model using LoRA (Low-Rank Adaptation).

**Key Parameters:**
- `--data`: Path to training data (CSV with 'prompt' and 'response' columns)
- `--output_dir`: Directory to save the trained model
- `--batch_size`: Start with 1-2 for GPUs with 8GB VRAM
- `--learning_rate`: Default 2e-4 works well for most cases
- `--src_lang`: Source language code (e.g., 'eng_Latn')
- `--tgt_lang`: Target language code (e.g., 'hin_Deva')

#### `test_model.py`
Tests the trained model with sample translations.

**Usage:**
```bash
python test_model.py --model_dir my_model --src_lang eng_Latn --tgt_lang hin_Deva
```

#### `generate_synthetic_data.py`
Generates additional training examples using back-translation.

**Usage:**
```bash
python generate_synthetic_data.py --input data.csv --output synthetic_data.csv --num_examples 1000
```

#### `inspect_dataset.py`
Displays dataset statistics and sample entries.

**Usage:**
```bash
python inspect_dataset.py --dataset_path hindi_combined --split train --num_samples 5
```

#### `filter_hindi_data.py`
Processes and filters the Hindi dataset from BPCC.

**Usage:**
```bash
python filter_hindi_data.py --output_dir data/processed
```

#### `load_bpcc.py`
Handles loading data from AI4Bharat's BPCC dataset.

**Features:**
- Downloads and processes BPCC dataset
- Handles language-specific splits
- Processes parallel corpora

**Usage:**
```python
from load_bpcc import load_bpcc_dataset
dataset = load_bpcc_dataset(split='train', language_pair=('en', 'hi'))
```

### Language Tags

This project uses standardized language tags:
- `eng_Latn`: English in Latin script
- `hin_Deva`: Hindi in Devanagari script

To change languages:
1. Update the language tags in your training command:
   ```bash
   python train_lora.py --src_lang eng_Latn --tgt_lang tam_Taml  # For English to Tamil
   ```
2. Ensure your dataset matches the language pair
3. Update the tokenizer configuration if needed

### Hugging Face Authentication

Some models require authentication:

1. **Get Your Token**
   - Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with 'read' access

2. **Using the Token**
   - **Option 1:** Set environment variable
     ```bash
     export HUGGINGFACE_HUB_TOKEN=your_token_here
     ```
   - **Option 2:** Login in Python
     ```python
     from huggingface_hub import login
     login(token='your_token_here')
     ```
   - **Option 3:** Use in terminal
     ```bash
     huggingface-cli login --token your_token_here
     ```

3. **In Your Code**
   ```python
   from transformers import AutoModel, AutoTokenizer
   
   model = AutoModel.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", 
                                   use_auth_token=True)  # Uses token from environment
   ```

### Data Processing Pipeline

1. **Raw Data**
   - BPCC dataset or custom parallel corpus
   - Expected format: Source and target language texts

2. **Preprocessing**
   - Normalization
   - Tokenization using language-specific tokenizers
   - Adding language tags

3. **Training**
   - Uses LoRA for efficient fine-tuning
   - Saves checkpoints during training

4. **Inference**
   - Loads the fine-tuned model
   - Handles input/output formatting
   - Manages language tags automatically

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in `train_lora.py`
   - Close other GPU applications
   - Add `--gradient_accumulation_steps 4` to train_lora.py

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a requirements.txt, install packages individually:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers==4.52.4 peft==0.15.2 datasets bitsandbytes
   ```

3. **CUDA Not Found**
   - Verify CUDA is installed: `nvcc --version`
   - Make sure your PyTorch version matches your CUDA version
   - Try reinstalling PyTorch with the correct CUDA version

4. **Slow Performance**
   - Ensure you're using a GPU (check with `nvidia-smi`)
   - Reduce batch size if using CPU
   - Close other memory-intensive applications

## 💡 Tips for Best Results

1. **Data Quality**
   - Ensure clean, grammatically correct translations
   - Include diverse sentence structures
   - At least 10,000 sentence pairs recommended

2. **Training**
   - Start with small batch size (1-2)
   - Train for at least 3 epochs
   - Monitor loss - it should decrease over time

3. **Hardware**
   - Use a GPU for training (CPU will be extremely slow)
   - More VRAM allows larger batch sizes
   - SSD storage helps with data loading speed

## 📚 Resources

- [AI4Bharat IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)

## 🤝 Need Help?

If you encounter any issues:
1. Check the error message carefully
2. Search online for the error
3. If still stuck, provide:
   - The exact error message
   - Your system specs
   - Steps to reproduce the issue

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- AI4Bharat for the IndicTrans2 model
- Hugging Face for the Transformers library
- PEFT for parameter-efficient fine-tuning
- The open-source community for their contributions

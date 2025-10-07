# Installation Guide - Python 3.8 Compatible

This project is fully compatible with Python 3.8 through 3.12.

## Quick Installation

### Using pip (Recommended)

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Using setup.py

```bash
# Install as a package
pip install -e .
```

## Python Version Compatibility

### Supported Python Versions
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12

### Verify Your Python Version

```bash
python --version
```

Expected output: `Python 3.8.x` or higher

## Step-by-Step Installation

### 1. Check Python Version

```bash
python --version
# or
python3 --version
```

If you don't have Python 3.8+, download from [python.org](https://www.python.org/downloads/)

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv vlm_env

# Activate on Linux/Mac
source vlm_env/bin/activate

# Activate on Windows
vlm_env\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- PyTorch 2.0+
- Transformers 4.37+
- All supporting libraries

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## GPU Support (CUDA)

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Install PyTorch with CUDA

If you need CUDA support, install PyTorch first:

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Then install other requirements:**
```bash
pip install -r requirements.txt --no-deps
pip install transformers pillow numpy pandas matplotlib seaborn accelerate datasets scikit-learn tqdm requests opencv-python sentencepiece protobuf einops timm scipy
```

## Package Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| transformers | ≥4.37.0 | Hugging Face models |
| pillow | ≥9.5.0 | Image processing |
| numpy | ≥1.20.0,<1.25.0 | Numerical computing |
| accelerate | ≥0.20.0 | Model optimization |

### Python 3.8 Specific Constraints

Some packages have version constraints for Python 3.8 compatibility:

- `numpy>=1.20.0,<1.25.0` - numpy 1.24+ requires Python 3.9+
- `pandas>=1.3.0,<2.1.0` - pandas 2.1+ requires Python 3.9+

These constraints ensure full compatibility with Python 3.8 while maintaining functionality.

## Common Installation Issues

### Issue: "No matching distribution found"

**Solution:** Ensure you're using Python 3.8+
```bash
python --version
```

### Issue: "ERROR: Could not find a version that satisfies the requirement"

**Solution:** Upgrade pip
```bash
pip install --upgrade pip
```

### Issue: PyTorch not detecting GPU

**Solution:** Reinstall PyTorch with CUDA support
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Microsoft Visual C++ 14.0 is required" (Windows)

**Solution:** Install Visual C++ Build Tools
- Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or use conda: `conda install -c conda-forge cxx-compiler`

### Issue: Out of memory during installation

**Solution:** Install packages one by one
```bash
pip install torch torchvision
pip install transformers
pip install pillow numpy pandas
# ... continue with remaining packages
```

## Alternative Installation Methods

### Using Conda

```bash
# Create conda environment
conda create -n vlm python=3.8

# Activate environment
conda activate vlm

# Install PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install -r requirements.txt
```

### Using Poetry

```bash
# Install poetry
pip install poetry

# Install dependencies
poetry install
```

### Using Docker

A Dockerfile is not provided, but you can create one:

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "image_captioning.py"]
```

## Testing Installation

### Quick Test

```bash
python -c "from model_loader import loader; print('✅ Installation successful!')"
```

### Full Test

```bash
python image_captioning.py
```

This will:
1. Load the model (first time takes 5-10 min)
2. Process a sample image
3. Generate a caption

## Minimal Installation

For testing or limited environments:

```bash
# Core only
pip install torch transformers pillow numpy requests
```

This installs just the essentials (2-3 GB vs full 10+ GB).

## Development Installation

If you plan to modify the code:

```bash
# Install in editable mode
pip install -e .

# Install development tools
pip install black flake8 pytest
```

## Uninstallation

### Remove packages

```bash
pip uninstall -r requirements.txt -y
```

### Remove virtual environment

```bash
# Deactivate first
deactivate

# Remove directory
rm -rf vlm_env
```

## Update Instructions

### Update all packages

```bash
pip install --upgrade -r requirements.txt
```

### Update specific package

```bash
pip install --upgrade transformers
```

## Requirements by Feature

### Basic Captioning
```
torch>=2.0.0
transformers>=4.37.0
pillow>=9.5.0
```

### Visual QA
```
torch>=2.0.0
transformers>=4.37.0
pillow>=9.5.0
```

### Fine-tuning
```
torch>=2.0.0
transformers>=4.37.0
accelerate>=0.20.0
datasets>=2.10.0
tqdm>=4.65.0
```

### Evaluation
```
numpy>=1.20.0,<1.25.0
scikit-learn>=1.0.0
pandas>=1.3.0,<2.1.0
```

## Platform-Specific Notes

### Linux
All packages install smoothly. Recommended platform.

### macOS
- Intel Macs: Full support
- Apple Silicon (M1/M2): Use MPS backend
  ```python
  # In config.py
  self.device = "mps" if torch.backends.mps.is_available() else "cpu"
  ```

### Windows
- Requires Visual C++ Build Tools
- Path length limitations may cause issues
  - Enable long paths in Windows
- Use PowerShell or Command Prompt

## Getting Help

1. **Check this guide first**
2. **Common issues section in README.md**
3. **Hugging Face forums**: https://discuss.huggingface.co/
4. **PyTorch forums**: https://discuss.pytorch.org/

## Next Steps

After successful installation:

1. **Quick Start**: Read `QUICKSTART.md`
2. **Run Example**: `python image_captioning.py`
3. **Full Learning**: Follow `LEARNING_GUIDE.md`

---

**Installation Status Checklist:**

- [ ] Python 3.8+ verified
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] GPU detected (if applicable)
- [ ] Test script runs successfully
- [ ] Ready to learn VLMs!

Enjoy your VLM journey! 🚀


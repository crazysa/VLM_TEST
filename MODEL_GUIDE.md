# Model Switching Guide

This guide shows you how to switch between different Vision-Language Models by only editing `config.py`.

## Current Model

**LLaVA 1.6 (Hermes 34B)**
```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
```

## How to Switch Models

### Step 1: Open config.py

### Step 2: Change the model_name

Replace line 8 with your desired model:

```python
self.model_name = "your-model-name-here"
```

### Step 3: Run any script

```bash
python image_captioning.py
python visual_qa.py
```

That's it! No other changes needed.

## Supported Models

### LLaVA Models (Recommended)

#### LLaVA 1.6 Hermes 34B (Current)
```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
```
- Size: 34B parameters
- VRAM: 24GB+ (or 8GB with 4-bit)
- Quality: Excellent
- Best for: Complex reasoning, detailed descriptions

#### LLaVA 1.6 Mistral 7B
```python
self.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
```
- Size: 7B parameters
- VRAM: 8GB+ (or 4GB with 4-bit)
- Quality: Very Good
- Best for: Balance of speed and quality

#### LLaVA 1.6 Vicuna 13B
```python
self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
```
- Size: 13B parameters
- VRAM: 16GB+ (or 6GB with 4-bit)
- Quality: Excellent
- Best for: Good balance

#### LLaVA 1.5 7B
```python
self.model_name = "llava-hf/llava-1.5-7b-hf"
```
- Size: 7B parameters
- VRAM: 8GB+
- Quality: Good
- Best for: Fast inference

#### LLaVA 1.5 13B
```python
self.model_name = "llava-hf/llava-1.5-13b-hf"
```
- Size: 13B parameters
- VRAM: 16GB+
- Quality: Very Good
- Best for: Detailed analysis

### BLIP-2 Models

#### BLIP-2 OPT 2.7B
```python
self.model_name = "Salesforce/blip2-opt-2.7b"
```
- Size: 2.7B parameters
- VRAM: 6GB+
- Quality: Good
- Best for: Fast captioning

#### BLIP-2 OPT 6.7B
```python
self.model_name = "Salesforce/blip2-opt-6.7b"
```
- Size: 6.7B parameters
- VRAM: 10GB+
- Quality: Very Good
- Best for: Better quality

#### BLIP-2 FlanT5 XL
```python
self.model_name = "Salesforce/blip2-flan-t5-xl"
```
- Size: 3B parameters
- VRAM: 8GB+
- Quality: Very Good
- Best for: Instruction following

#### BLIP-2 FlanT5 XXL
```python
self.model_name = "Salesforce/blip2-flan-t5-xxl"
```
- Size: 11B parameters
- VRAM: 16GB+
- Quality: Excellent
- Best for: Best BLIP-2 quality

### Other Models

#### InstructBLIP Vicuna 7B
```python
self.model_name = "Salesforce/instructblip-vicuna-7b"
```
- Size: 7B parameters
- VRAM: 8GB+
- Quality: Good
- Best for: Instruction following

#### InstructBLIP Vicuna 13B
```python
self.model_name = "Salesforce/instructblip-vicuna-13b"
```
- Size: 13B parameters
- VRAM: 16GB+
- Quality: Very Good
- Best for: Complex instructions

## Memory Optimization

### For Limited VRAM

Add these to config.py:

```python
self.load_in_4bit = True
self.batch_size = 1
```

This reduces VRAM usage by ~75%

### For 8GB VRAM

```python
self.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
self.load_in_4bit = True
self.batch_size = 1
```

### For 16GB VRAM

```python
self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
self.load_in_4bit = True
self.batch_size = 2
```

### For 24GB+ VRAM

```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
self.load_in_8bit = True
self.batch_size = 4
```

### For 40GB+ VRAM

```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
self.torch_dtype = torch.float16
self.batch_size = 8
```

## Generation Quality Settings

### For Best Quality

```python
self.temperature = 0.7
self.top_p = 0.9
self.top_k = 50
self.do_sample = True
self.max_new_tokens = 500
```

### For Fastest Speed

```python
self.temperature = 1.0
self.do_sample = False
self.max_new_tokens = 100
```

### For Factual Responses

```python
self.temperature = 0.1
self.top_p = 0.95
self.do_sample = True
```

### For Creative Responses

```python
self.temperature = 1.0
self.top_p = 0.95
self.top_k = 100
self.do_sample = True
```

## Complete Config Examples

### Budget Setup (8GB VRAM)

```python
self.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
self.batch_size = 1
self.load_in_4bit = True
self.max_new_tokens = 200
self.temperature = 0.7
```

### Balanced Setup (16GB VRAM)

```python
self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
self.batch_size = 2
self.load_in_8bit = True
self.max_new_tokens = 300
self.temperature = 0.7
```

### High-End Setup (24GB+ VRAM)

```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
self.batch_size = 4
self.torch_dtype = torch.float16
self.max_new_tokens = 500
self.temperature = 0.7
```

### Production Setup (Speed Priority)

```python
self.model_name = "Salesforce/blip2-flan-t5-xl"
self.batch_size = 8
self.torch_dtype = torch.float16
self.max_new_tokens = 150
self.do_sample = False
```

## Testing Different Models

### Quick Test Script

```python
from image_captioning import ImageCaptioner

captioner = ImageCaptioner()
test_image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
result = captioner.generate_caption(test_image)
print(result)
```

Run this after each model change to verify it works.

## Troubleshooting

### Out of Memory Error
- Enable 4-bit quantization: `self.load_in_4bit = True`
- Reduce batch size: `self.batch_size = 1`
- Use smaller model

### Model Not Found
- Check model name on Hugging Face
- Ensure you have internet connection
- Clear cache: `rm -rf model_cache/`

### Slow Generation
- Disable sampling: `self.do_sample = False`
- Reduce max tokens: `self.max_new_tokens = 100`
- Use smaller model

### Poor Quality
- Increase temperature: `self.temperature = 0.8`
- Increase max tokens: `self.max_new_tokens = 500`
- Use larger model

## Benchmarks

| Model | VRAM (FP16) | VRAM (4-bit) | Speed | Quality |
|-------|-------------|--------------|-------|---------|
| LLaVA 1.6 34B | 70GB | 20GB | Slow | ⭐⭐⭐⭐⭐ |
| LLaVA 1.6 13B | 28GB | 8GB | Medium | ⭐⭐⭐⭐ |
| LLaVA 1.6 7B | 16GB | 5GB | Fast | ⭐⭐⭐⭐ |
| LLaVA 1.5 13B | 28GB | 8GB | Medium | ⭐⭐⭐⭐ |
| LLaVA 1.5 7B | 16GB | 5GB | Fast | ⭐⭐⭐ |
| BLIP-2 T5 XXL | 24GB | 7GB | Fast | ⭐⭐⭐⭐ |
| BLIP-2 T5 XL | 8GB | 3GB | Very Fast | ⭐⭐⭐ |
| BLIP-2 OPT 6.7B | 14GB | 5GB | Fast | ⭐⭐⭐ |
| BLIP-2 OPT 2.7B | 6GB | 2GB | Very Fast | ⭐⭐⭐ |

## Recommendations

### For Research
- LLaVA 1.6 34B (Hermes)
- High quality, excellent reasoning

### For Production
- LLaVA 1.6 Mistral 7B
- Good balance of speed and quality

### For Development/Testing
- BLIP-2 FlanT5 XL
- Fast iteration, good quality

### For Edge Devices
- BLIP-2 OPT 2.7B
- Minimal requirements

## Next Steps

1. Choose your model from the list above
2. Update `config.py` with the model name
3. Adjust memory settings if needed
4. Run `python example_usage.py`
5. Experiment with generation parameters

Remember: **Only config.py needs to be changed!**


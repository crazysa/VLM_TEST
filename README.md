# Vision-Language Models (VLM) - Universal Implementation

A flexible, model-agnostic implementation for Vision-Language Models that works with any Hugging Face VLM by simply changing the model name in `config.py`.

## Currently Configured Model

**LLaVA 1.6 (Hermes 34B)** - `NousResearch/llava-v1.6-34b-hf`

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Change Model (Optional)

Edit `config.py` and change the `model_name`:

```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
```

Supported models include:
- `NousResearch/llava-v1.6-34b-hf` (LLaVA 1.6 Hermes 34B)
- `llava-hf/llava-1.5-7b-hf` (LLaVA 1.5 7B)
- `llava-hf/llava-1.5-13b-hf` (LLaVA 1.5 13B)
- `Salesforce/blip2-opt-2.7b` (BLIP-2)
- `Salesforce/blip2-flan-t5-xl` (BLIP-2 with T5)
- Any Hugging Face VLM with AutoProcessor support

### 3. Run Any Task

```bash
python image_captioning.py
python visual_qa.py
python image_analysis.py
python conversational_vlm.py
python multi_image_reasoning.py
```

## Architecture

### Universal Model Loading

The `model_loader.py` automatically handles:
- AutoProcessor detection and fallback
- Model architecture detection (LLaVA, BLIP, etc.)
- Device placement (CPU/CUDA)
- Quantization (8-bit/4-bit)
- Generation configuration

### Single Source of Truth

All configuration is in `config.py`:
- Model selection
- Generation parameters
- Training hyperparameters
- Device settings
- Quantization options

## File Structure

```
VLM_TEST/
├── config.py                    # ALL configuration here
├── model_loader.py              # Universal model loader
├── image_captioning.py          # Generate image descriptions
├── visual_qa.py                 # Answer questions about images
├── image_analysis.py            # Detailed image analysis
├── multi_image_reasoning.py    # Compare/analyze multiple images
├── conversational_vlm.py        # Multi-turn conversations
├── fine_tuning.py              # Fine-tune on custom data
├── evaluation.py               # Evaluate model performance
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Usage Examples

### Image Captioning

```python
from image_captioning import ImageCaptioner

captioner = ImageCaptioner()
caption = captioner.generate_caption("image.jpg")
print(caption)
```

### Visual Question Answering

```python
from visual_qa import VisualQA

vqa = VisualQA()
answer = vqa.answer_question("image.jpg", "What is in this image?")
print(answer)
```

### Image Analysis

```python
from image_analysis import ImageAnalyzer

analyzer = ImageAnalyzer()
objects = analyzer.detect_objects("image.jpg")
scene = analyzer.describe_scene("image.jpg")
colors = analyzer.identify_colors("image.jpg")
```

### Conversational VLM

```python
from conversational_vlm import ConversationalVLM

vlm = ConversationalVLM()
response1 = vlm.chat("image.jpg", "What do you see?", reset_history=True)
response2 = vlm.chat("image.jpg", "What color is it?")
```

### Multi-Image Reasoning

```python
from multi_image_reasoning import MultiImageReasoner

reasoner = MultiImageReasoner()
comparison = reasoner.compare_images("image1.jpg", "image2.jpg")
```

### Interactive Mode

```bash
python visual_qa.py
python conversational_vlm.py
```

## Configuration Options

### Model Settings

```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
self.model_type = "llava"
```

### Generation Parameters

```python
self.max_new_tokens = 200
self.temperature = 0.7
self.top_p = 0.9
self.top_k = 50
self.do_sample = True
```

### Memory Optimization

```python
self.load_in_8bit = False
self.load_in_4bit = True
self.torch_dtype = torch.float16
```

### Training Parameters

```python
self.learning_rate = 2e-5
self.num_epochs = 3
self.batch_size = 4
self.gradient_accumulation_steps = 4
```

## Switching Models

### To LLaVA 1.5 7B

```python
self.model_name = "llava-hf/llava-1.5-7b-hf"
```

### To BLIP-2

```python
self.model_name = "Salesforce/blip2-opt-2.7b"
```

### To Any Other Model

```python
self.model_name = "your-model-name-here"
```

That's it! No other changes needed.

## Prompt Format

The implementation automatically handles LLaVA-style prompts:

```
USER: <image>
{your question or instruction}
ASSISTANT:
```

For other models, the format is automatically adapted.

## Fine-Tuning

### Prepare Dataset

Create a JSON file:

```json
[
  {
    "image_path": "path/to/image1.jpg",
    "question": "What is in this image?",
    "answer": "A detailed description."
  }
]
```

### Train

```python
from fine_tuning import VLMFineTuner

trainer = VLMFineTuner()
trainer.train("data/train.json", num_epochs=3)
```

## Evaluation

```python
from evaluation import VLMEvaluator

evaluator = VLMEvaluator()
results = evaluator.evaluate_generation(predictions, references)
evaluator.print_summary()
```

## Performance Tips

### For Large Models (34B)

```python
self.load_in_4bit = True
self.batch_size = 1
self.gradient_accumulation_steps = 8
```

### For Speed

```python
self.torch_dtype = torch.float16
self.batch_size = 8
self.do_sample = False
```

### For Quality

```python
self.temperature = 0.7
self.top_p = 0.9
self.num_beams = 5
self.do_sample = True
```

## System Requirements

### Minimum

- Python 3.8+
- 16GB RAM
- CUDA-capable GPU with 8GB VRAM

### Recommended for LLaVA 34B

- 32GB+ RAM
- CUDA GPU with 24GB+ VRAM
- Or use 4-bit quantization

## Troubleshooting

### Out of Memory

```python
config.load_in_4bit = True
config.batch_size = 1
```

### Slow Generation

```python
config.do_sample = False
config.num_beams = 1
```

### Poor Quality

```python
config.temperature = 0.7
config.max_new_tokens = 500
```

## Advanced Features

### Custom Prompts

```python
captioner = ImageCaptioner()
custom_prompt = "USER: <image>\nProvide a poetic description.\nASSISTANT:"
result = captioner.generate_caption("image.jpg", prompt=custom_prompt)
```

### Batch Processing

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
captions = captioner.batch_caption(images)
```

### Multi-Turn Conversations

```python
vlm = ConversationalVLM()
vlm.chat("image.jpg", "What's in this image?", reset_history=True)
vlm.chat("image.jpg", "Tell me more about the colors")
vlm.chat("image.jpg", "What's the mood?")
history = vlm.get_history()
```

## Model Comparison

| Model | Size | Speed | Quality | VRAM |
|-------|------|-------|---------|------|
| LLaVA 1.6 34B | 34B | Slow | Excellent | 24GB+ |
| LLaVA 1.5 13B | 13B | Medium | Very Good | 16GB |
| LLaVA 1.5 7B | 7B | Fast | Good | 8GB |
| BLIP-2 | 2.7B | Very Fast | Good | 6GB |

## License

MIT License

## Citation

If using LLaVA:
```
@misc{liu2023llava,
    title={Visual Instruction Tuning},
    author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    year={2023}
}
```

## Support

For issues or questions:
1. Check configuration in `config.py`
2. Verify model name is correct on Hugging Face
3. Check VRAM requirements
4. Try 4-bit quantization

## Contributing

This is a learning repository. To add support for new models:
1. Update `model_loader.py` if needed
2. Test with the new model
3. Update this README

Happy Learning!


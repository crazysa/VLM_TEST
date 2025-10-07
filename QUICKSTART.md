# Quick Start Guide - Learn VLMs in 1 Hour

Get started with Vision-Language Models immediately. No theory, just hands-on practice.

## Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Model

Open `config.py` and select based on your GPU:

**For 8GB GPU (or want to start fast):**
```python
self.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
self.load_in_4bit = True
self.batch_size = 1
```

**For 16GB+ GPU:**
```python
self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
self.load_in_4bit = True
self.batch_size = 2
```

**For 24GB+ GPU (best quality):**
```python
self.model_name = "NousResearch/llava-v1.6-34b-hf"
self.load_in_8bit = True
self.batch_size = 4
```

## Your First 5 Tasks (55 minutes)

### Task 1: Image Captioning (10 min)

```bash
python image_captioning.py
```

This will:
1. Load the model (first time takes 5-10 min)
2. Download a sample image
3. Generate a description

**Modify it**: Edit `image_captioning.py`, line 72:
```python
sample_image = "YOUR_IMAGE_URL_OR_PATH"
```

Run again!

### Task 2: Ask Questions (10 min)

```bash
python visual_qa.py
```

**Try these questions** (edit line 76-80):
```python
questions = [
    "What's the main object?",
    "What color is it?",
    "Where is this photo taken?",
    "What time of day is it?",
    "What's the mood?"
]
```

### Task 3: Detailed Analysis (10 min)

```bash
python image_analysis.py
```

This will:
- Detect objects
- Describe the scene
- Identify colors

**Custom analysis** (edit line 69):
```python
print("\nCustom Analysis:")
custom = analyzer.analyze(sample_image, "YOUR QUESTION HERE")
print(custom)
```

### Task 4: Have a Conversation (10 min)

```bash
python conversational_vlm.py
```

For interactive mode, edit the `main()` function:
```python
vlm = ConversationalVLM()
sample_image = "your_image.jpg"
vlm.interactive_mode(sample_image)
```

Then run and type your questions!

### Task 5: Compare Images (15 min)

```bash
python multi_image_reasoning.py
```

**Compare your own images**:
```python
from multi_image_reasoning import MultiImageReasoner

reasoner = MultiImageReasoner()
comparison = reasoner.compare_images(
    "image1.jpg",
    "image2.jpg",
    "What changed between these two images?"
)
print(comparison)
```

## What You Just Learned

✅ How to load and run VLMs  
✅ Image captioning  
✅ Visual question answering  
✅ Multi-turn conversations  
✅ Multi-image reasoning  

## Next Steps (Choose One)

### Path A: Build Something Useful

Pick a 15-minute project:

**Project: Photo Organizer**
```python
from image_analysis import ImageAnalyzer
from pathlib import Path
import json

analyzer = ImageAnalyzer()
photos = Path("YOUR_PHOTOS_FOLDER")
categories = {}

for photo in photos.glob("*.jpg"):
    print(f"Analyzing {photo.name}...")
    category = analyzer.analyze(
        str(photo),
        "What's the main subject? Answer in 1-2 words."
    )
    
    if category not in categories:
        categories[category] = []
    categories[category].append(photo.name)

print(json.dumps(categories, indent=2))
```

### Path B: Experiment with Parameters

Try different settings in `config.py`:

**For creative descriptions:**
```python
self.temperature = 1.2
self.max_new_tokens = 500
```

**For factual answers:**
```python
self.temperature = 0.1
self.max_new_tokens = 50
```

### Path C: Try Different Models

Switch models by editing `config.py` line 8:

```python
self.model_name = "Salesforce/blip2-flan-t5-xl"
```

Run the same tasks again and compare!

## Common First-Time Issues

### "Out of memory"
```python
self.load_in_4bit = True
self.batch_size = 1
```

### "Too slow"
```python
self.model_name = "Salesforce/blip2-opt-2.7b"
self.do_sample = False
```

### "Poor quality responses"
```python
self.temperature = 0.7
self.max_new_tokens = 300
self.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
```

## Your First Real Project (30 minutes)

Build an image description tool:

```python
from image_captioning import ImageCaptioner
from image_analysis import ImageAnalyzer
import sys

def describe_image(image_path):
    captioner = ImageCaptioner()
    analyzer = ImageAnalyzer()
    
    print("Analyzing image...\n")
    
    print("=== CAPTION ===")
    print(captioner.generate_caption(image_path))
    
    print("\n=== OBJECTS ===")
    print(analyzer.detect_objects(image_path))
    
    print("\n=== COLORS ===")
    print(analyzer.identify_colors(image_path))
    
    print("\n=== SCENE ===")
    print(analyzer.describe_scene(image_path))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        describe_image(sys.argv[1])
    else:
        print("Usage: python describe.py <image_path>")
```

Save as `describe.py`, then use:
```bash
python describe.py my_photo.jpg
```

## Explore More

Ready to go deeper? Open `LEARNING_GUIDE.md` for:
- 6-week structured curriculum
- 20+ complete projects
- Advanced techniques
- Production deployment

Or continue experimenting on your own!

## Quick Reference

### Load a Different Model
```python
# Edit config.py line 8
self.model_name = "MODEL_NAME_HERE"
```

### Adjust Quality
```python
# Edit config.py lines 15-21
self.temperature = 0.7  # 0.1 (factual) to 1.5 (creative)
self.max_new_tokens = 200  # 50 (short) to 500 (long)
```

### Save Memory
```python
# Edit config.py lines 27-28
self.load_in_4bit = True
self.batch_size = 1
```

### Custom Prompts
```python
prompt = "USER: <image>\nYOUR_INSTRUCTION_HERE\nASSISTANT:"
result = captioner.generate_caption(image_path, prompt=prompt)
```

## Resources

- **Full Guide**: `LEARNING_GUIDE.md` - Complete 6-week curriculum
- **Model Guide**: `MODEL_GUIDE.md` - All available models
- **Examples**: `example_usage.py` - Interactive demos
- **README**: `README.md` - Technical documentation

## Get Help

1. Check error message
2. Try suggested fixes above
3. Read full documentation
4. Check Hugging Face model page

## Congratulations!

You've completed the quickstart. You now know how to:
- Run VLMs
- Generate captions
- Ask questions about images
- Analyze images in detail
- Compare multiple images

Keep building! 🚀


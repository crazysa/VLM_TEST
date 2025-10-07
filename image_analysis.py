from PIL import Image
from model_loader import loader
from config import config
import requests
from io import BytesIO
import torch

class ImageAnalyzer:
    def __init__(self):
        self.processor, self.model = loader.load_model()
        self.device = config.device
        self.gen_config = loader.get_generation_config()
        
    def load_image(self, image_path):
        if image_path.startswith('http'):
            response = requests.get(image_path, timeout=30)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    
    def analyze(self, image_path, prompt):
        image = self.load_image(image_path)
        
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        )
        
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self.device)
        else:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_config)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response
    
    def detect_objects(self, image_path):
        prompt = "List all the objects you can see in this image."
        return self.analyze(image_path, prompt)
    
    def describe_scene(self, image_path):
        prompt = "Describe the scene in this image in detail, including objects, people, actions, and the setting."
        return self.analyze(image_path, prompt)
    
    def identify_colors(self, image_path):
        prompt = "What are the dominant colors in this image?"
        return self.analyze(image_path, prompt)
    
    def count_objects(self, image_path, object_name):
        prompt = f"How many {object_name} are in this image?"
        return self.analyze(image_path, prompt)
    
    def read_text(self, image_path):
        prompt = "Read and transcribe any text visible in this image."
        return self.analyze(image_path, prompt)
    
    def classify_image(self, image_path, categories):
        categories_str = ", ".join(categories)
        prompt = f"Which of these categories best describes this image: {categories_str}? Respond with just the category name."
        return self.analyze(image_path, prompt)

def main():
    analyzer = ImageAnalyzer()
    
    # sample_image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    sample_image = "/home/shub/Pictures/Screenshots/Screenshot from 2025-10-07 15-48-57.png"
    
    print("Object Detection:")
    print(analyzer.detect_objects(sample_image))
    
    print("\nScene Description:")
    print(analyzer.describe_scene(sample_image))
    
    print("\nColor Identification:")
    print(analyzer.identify_colors(sample_image))

if __name__ == "__main__":
    main()


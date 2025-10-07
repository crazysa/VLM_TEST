from PIL import Image
from model_loader import loader
from config import config
import requests
from io import BytesIO
import torch

class MultiImageReasoner:
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
    
    def compare_images(self, image_path1, image_path2, question=None):
        image1 = self.load_image(image_path1)
        image2 = self.load_image(image_path2)
        
        if question is None:
            question = "What are the similarities and differences between these two images?"
        
        prompt = f"USER: <image>\n<image>\n{question}\nASSISTANT:"
        
        try:
            inputs = self.processor(
                text=prompt,
                images=[image1, image2],
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
        except Exception:
            answer1 = self._analyze_single(image_path1, f"Describe this image. {question}")
            answer2 = self._analyze_single(image_path2, f"Describe this image. {question}")
            return f"Image 1: {answer1}\n\nImage 2: {answer2}"
    
    def _analyze_single(self, image_path, prompt):
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
    
    def sequence_analysis(self, image_paths, question):
        results = []
        for i, path in enumerate(image_paths):
            prompt = f"{question} (Image {i+1} of {len(image_paths)})"
            result = self._analyze_single(path, prompt)
            results.append(f"Image {i+1}: {result}")
        
        return "\n\n".join(results)

def main():
    reasoner = MultiImageReasoner()
    
    sample_image1 = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    sample_image2 = "https://images.unsplash.com/photo-1543466835-00a7907e9de1"
    
    comparison = reasoner.compare_images(sample_image1, sample_image2)
    print("Image Comparison:")
    print(comparison)

if __name__ == "__main__":
    main()


from PIL import Image
from model_loader import loader
from config import config
import requests
from io import BytesIO
import torch

class ImageCaptioner:
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
    
    def generate_caption(self, image_path, prompt=None):
        image = self.load_image(image_path)
        
        if prompt is None:
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        
        inputs = self.processor(
            text=prompt,
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
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()
        
        return caption
    
    def batch_caption(self, image_paths, prompts=None):
        images = [self.load_image(path) for path in image_paths]
        
        if prompts is None:
            prompts = ["USER: <image>\nDescribe this image in detail.\nASSISTANT:"] * len(images)
        
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self.device)
        else:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_config)
        
        captions = []
        for output in outputs:
            caption = self.processor.decode(output, skip_special_tokens=True)
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[-1].strip()
            captions.append(caption)
        
        return captions

def main():
    captioner = ImageCaptioner()
    
    sample_image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    caption = captioner.generate_caption(sample_image)
    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()


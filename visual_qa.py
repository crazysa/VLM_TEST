from PIL import Image
from model_loader import loader
from config import config
import requests
from io import BytesIO
import torch

class VisualQA:
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
    
    def answer_question(self, image_path, question):
        image = self.load_image(image_path)
        
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
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
        
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()
        
        return answer
    
    def batch_qa(self, image_paths, questions):
        images = [self.load_image(path) for path in image_paths]
        
        prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
        
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
        
        answers = []
        for output in outputs:
            answer = self.processor.decode(output, skip_special_tokens=True)
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[-1].strip()
            answers.append(answer)
        
        return answers
    
    def interactive_qa(self, image_path):
        print("Interactive VQA Mode - Type 'quit' to exit")
        while True:
            question = input("Question: ")
            if question.lower() == 'quit':
                break
            
            answer = self.answer_question(image_path, question)
            print(f"Answer: {answer}\n")

def main():
    vqa = VisualQA()
    
    sample_image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    question = "What animal is in the image?"
    answer = vqa.answer_question(sample_image, question)
    print(f"Q: {question}")
    print(f"A: {answer}")

if __name__ == "__main__":
    main()


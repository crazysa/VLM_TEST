from PIL import Image
from model_loader import loader
from config import config
import requests
from io import BytesIO
import torch

class ConversationalVLM:
    def __init__(self):
        self.processor, self.model = loader.load_model()
        self.device = config.device
        self.gen_config = loader.get_generation_config()
        self.conversation_history = []
        
    def load_image(self, image_path):
        if image_path.startswith('http'):
            response = requests.get(image_path, timeout=30)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    
    def chat(self, image_path, user_message, reset_history=False):
        if reset_history:
            self.conversation_history = []
        
        image = self.load_image(image_path)
        
        self.conversation_history.append("USER: " + user_message)
        
        if len(self.conversation_history) == 1:
            prompt = f"USER: <image>\n{user_message}\nASSISTANT:"
        else:
            conversation_text = "\n".join(self.conversation_history)
            prompt = f"USER: <image>\n{conversation_text}\nASSISTANT:"
        
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
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        self.conversation_history.append("ASSISTANT: " + response)
        
        return response
    
    def interactive_mode(self, image_path):
        print("Conversational VLM - Type 'quit' to exit, 'reset' to clear history")
        print("=" * 60)
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reset':
                self.conversation_history = []
                print("Conversation history cleared.")
                continue
            
            response = self.chat(image_path, user_input)
            print(f"Assistant: {response}")
    
    def get_history(self):
        return self.conversation_history
    
    def clear_history(self):
        self.conversation_history = []

def main():
    vlm = ConversationalVLM()
    
    sample_image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    
    response1 = vlm.chat(sample_image, "What do you see in this image?", reset_history=True)
    print(f"Q1: What do you see in this image?")
    print(f"A1: {response1}\n")
    
    response2 = vlm.chat(sample_image, "What color is it?")
    print(f"Q2: What color is it?")
    print(f"A2: {response2}")

if __name__ == "__main__":
    main()


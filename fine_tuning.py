from PIL import Image
from model_loader import loader
from config import config
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from pathlib import Path

class VLMDataset(Dataset):
    def __init__(self, data_file, processor):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        prompt = f"USER: <image>\n{item['question']}\nASSISTANT: {item['answer']}"
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        return {k: v.squeeze(0) for k, v in inputs.items()}

class VLMFineTuner:
    def __init__(self):
        self.processor, self.model = loader.load_model()
        self.device = config.device
        
    def prepare_data(self, train_file, batch_size=None):
        if batch_size is None:
            batch_size = config.batch_size
        
        dataset = VLMDataset(train_file, self.processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        return dataloader
    
    def train(self, train_file, num_epochs=None, save_dir=None):
        if num_epochs is None:
            num_epochs = config.num_epochs
        
        if save_dir is None:
            save_dir = config.output_dir / "fine_tuned_model"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        dataloader = self.prepare_data(train_file)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            
            progress_bar = tqdm(dataloader, desc="Training")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss: {avg_loss:.4f}")
            
            self.save_model(save_dir / f"checkpoint_epoch_{epoch+1}")
        
        return save_dir
    
    def save_model(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

def create_sample_dataset():
    sample_data = [
        {
            "image_path": "path/to/image1.jpg",
            "question": "What is in this image?",
            "answer": "A detailed description of the image."
        },
        {
            "image_path": "path/to/image2.jpg",
            "question": "Describe the scene.",
            "answer": "Another detailed description."
        }
    ]
    
    output_file = config.data_dir / "sample_train.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample dataset created at: {output_file}")

def main():
    print("Fine-tuning requires a dataset in JSON format.")
    print("Each entry should have 'image_path', 'question', and 'answer' fields.")
    create_sample_dataset()

if __name__ == "__main__":
    main()


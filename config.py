import torch
from pathlib import Path

class VLMConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = "NousResearch/llava-v1.6-34b-hf"
        
        self.model_type = "llava"
        
        self.cache_dir = Path("./model_cache")
        self.output_dir = Path("./outputs")
        self.data_dir = Path("./data")
        
        self.batch_size = 4
        self.max_length = 512
        self.max_new_tokens = 200
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.num_beams = 1
        self.do_sample = True
        
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 4
        
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.seed = 42
        
    def create_directories(self):
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)

config = VLMConfig()
config.create_directories()


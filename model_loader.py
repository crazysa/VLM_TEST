from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from config import config
import torch

class VLMLoader:
    def __init__(self):
        self.config = config
        self.processor = None
        self.model = None
        
    def load_model(self):
        model_kwargs = {
            'cache_dir': self.config.cache_dir,
            'torch_dtype': self.config.torch_dtype,
        }
        
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        elif self.config.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = 'auto'
        elif self.config.device == 'cuda':
            model_kwargs['device_map'] = 'auto'
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        return self.processor, self.model
    
    def get_generation_config(self):
        gen_config = {
            'max_new_tokens': self.config.max_new_tokens,
            'do_sample': self.config.do_sample,
        }
        
        if self.config.do_sample:
            gen_config['temperature'] = self.config.temperature
            gen_config['top_p'] = self.config.top_p
            if self.config.top_k is not None:
                gen_config['top_k'] = self.config.top_k
        
        if self.config.num_beams > 1:
            gen_config['num_beams'] = self.config.num_beams
            
        return gen_config

loader = VLMLoader()

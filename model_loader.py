from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, LlavaForConditionalGeneration
from config import config

class VLMLoader:
    def __init__(self):
        self.config = config
        self.processor = None
        self.model = None
        
    def load_model(self):
        model_kwargs = {
            'cache_dir': self.config.cache_dir,
            'torch_dtype': self.config.torch_dtype,
            'device_map': 'auto' if self.config.device == 'cuda' else None,
        }
        
        if self.config.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif self.config.load_in_4bit:
            model_kwargs['load_in_4bit'] = True
            
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
        except Exception as e:
            print(f"Failed to load with AutoProcessor: {e}")
            from transformers import LlavaProcessor
            self.processor = LlavaProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
        
        try:
            if 'llava' in self.config.model_name.lower():
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
        
        if not model_kwargs.get('device_map') and self.config.device == 'cuda':
            self.model = self.model.to(self.config.device)
        
        return self.processor, self.model
    
    def get_generation_config(self):
        gen_config = {
            'max_new_tokens': self.config.max_new_tokens,
            'do_sample': self.config.do_sample,
        }
        
        if self.config.do_sample:
            gen_config.update({
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
            })
        
        if self.config.num_beams > 1:
            gen_config['num_beams'] = self.config.num_beams
            
        return gen_config

loader = VLMLoader()


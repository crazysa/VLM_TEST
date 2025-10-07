import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
import json
from pathlib import Path

class VLMEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def compute_bleu_score(self, predictions, references):
        from collections import Counter
        
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            scores = []
            for n in range(1, 5):
                pred_ngrams = Counter(get_ngrams(pred_tokens, n))
                ref_ngrams = Counter(get_ngrams(ref_tokens, n))
                
                matches = sum((pred_ngrams & ref_ngrams).values())
                total = max(len(pred_tokens) - n + 1, 1)
                
                scores.append(matches / total if total > 0 else 0)
            
            bleu_scores.append(np.mean(scores))
        
        return np.mean(bleu_scores)
    
    def compute_exact_match(self, predictions, references):
        matches = [
            1 if pred.lower().strip() == ref.lower().strip() else 0
            for pred, ref in zip(predictions, references)
        ]
        return np.mean(matches)
    
    def compute_accuracy(self, predictions, references):
        return accuracy_score(references, predictions)
    
    def evaluate_generation(self, predictions, references):
        bleu = self.compute_bleu_score(predictions, references)
        exact_match = self.compute_exact_match(predictions, references)
        
        results = {
            'bleu': bleu,
            'exact_match': exact_match,
            'num_samples': len(predictions)
        }
        
        self.metrics['generation'].append(results)
        return results
    
    def evaluate_classification(self, predictions, references):
        accuracy = self.compute_accuracy(predictions, references)
        
        results = {
            'accuracy': accuracy,
            'num_samples': len(predictions)
        }
        
        self.metrics['classification'].append(results)
        return results
    
    def save_results(self, output_file):
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        results = dict(self.metrics)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def print_summary(self):
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for task, results_list in self.metrics.items():
            print(f"\n{task.upper()}:")
            if not results_list:
                continue
            
            latest = results_list[-1]
            for metric, value in latest.items():
                if metric != 'num_samples':
                    print(f"  {metric}: {value:.4f}")
            
            if 'num_samples' in latest:
                print(f"  samples: {latest['num_samples']}")

def main():
    evaluator = VLMEvaluator()
    
    predictions = [
        "a cat sitting on a couch",
        "a dog playing in the park"
    ]
    references = [
        "a cat on a sofa",
        "a dog in a park"
    ]
    
    evaluator.evaluate_generation(predictions, references)
    evaluator.print_summary()

if __name__ == "__main__":
    main()


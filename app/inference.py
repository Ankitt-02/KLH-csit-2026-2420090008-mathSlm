import argparse
import os
import yaml
import torch

from model.config import MathSLMConfig
from model.transformer import MathSLM
from model.generation import generate, parse_generated_text, generate_self_consistency
from tokenizer.tokenizer import MathTokenizer

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class MathSLMInferenceEngine:
    def __init__(self, config_path="config/config.yaml", checkpoint_path=None):
        self.config = load_config(config_path)
        self.device = "cpu" if self.config.get("smoke_test", False) else self.config["training"].get("device", "cpu")

        tokenizer_path = self.config["paths"]["tokenizer"]
        self.tokenizer = MathTokenizer.load(tokenizer_path)

        self.config["model"]["vocab_size"] = self.tokenizer.vocab_size
        model_config = MathSLMConfig.from_dict(self.config["model"])
        self.model = MathSLM(model_config)

        if checkpoint_path is None:
            ckpt_dir = self.config["paths"]["checkpoints"]
            best_ckpt = os.path.join(ckpt_dir, "model_best.pt")
            final_ckpt = os.path.join(ckpt_dir, "model_final.pt")
            if os.path.exists(best_ckpt):
                checkpoint_path = best_ckpt
            elif os.path.exists(final_ckpt):
                checkpoint_path = final_ckpt

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print(f"Loaded trained MathSLM weights from {checkpoint_path}")
        else:
            print("Warning: No checkpoint loaded. Operating with initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def solve(self, question: str, self_consistency: bool = False, temperature: float = 0.0, max_new_tokens: int = 256):
        prompt = f"[Q] {question} [R]"
        if self_consistency:
            best_sample, samples, stats = generate_self_consistency(
                self.model, self.tokenizer, prompt, num_samples=5, max_new_tokens=max_new_tokens, temperature=0.7, device=self.device
            )
            return {
                "question": question,
                "reasoning": best_sample["reasoning"],
                "answer": best_sample["answer"],
                "vote_stats": stats
            }
        else:
            input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
            out_ids = generate(self.model, input_ids, max_new_tokens=max_new_tokens, temperature=temperature, device=self.device, eos_id=self.tokenizer.eos_id)
            out_text = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)
            reasoning, answer = parse_generated_text(out_text)
            return {
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
                "raw_output": out_text
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--self_consistency", action="store_true")
    args = parser.parse_args()

    engine = MathSLMInferenceEngine(config_path=args.config)
    result = engine.solve(args.question, self_consistency=args.self_consistency)

    print("\n" + "="*50)
    print(f"Question: {result['question']}")
    print("-" * 50)
    print(f"Reasoning:\n{result['reasoning']}")
    print("-" * 50)
    print(f"Final Stated Answer: {result['answer']}")
    print("="*50)

if __name__ == "__main__":
    main()

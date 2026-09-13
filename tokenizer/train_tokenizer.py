import argparse
import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_from_disk
from tokenizers import Tokenizer, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Metaspace, Digits, Split

from tokenizer.tokenizer import MathTokenizer

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_training_corpus(dataset_dict, batch_size=1000):
    for split in ["train", "val"]:
        if split in dataset_dict:
            dataset = dataset_dict[split]
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                for q, r, a in zip(batch["question"], batch["reasoning"], batch["answer"]):
                    yield f"[Q] {q} [R] {r} [A] {a} [EOS]"

def build_and_train_tokenizer(dataset_dict, target_vocab_size=16384):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    pre_toks = Sequence([
        Metaspace(replacement=" ", prepend_scheme="never"),
        Digits(individual_digits=True),
        Split(Regex(r"\\[a-zA-Z]+"), behavior="isolated")
    ])
    tokenizer.pre_tokenizer = pre_toks
    tokenizer.decoder = decoders.Metaspace(replacement=" ", prepend_scheme="never")

    special_tokens = ["[PAD]", "[UNK]", "[EOS]", "[Q]", "[R]", "[A]"]

    trainer = BpeTrainer(
        vocab_size=target_vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2
    )

    print(f"Training mathematical BPE tokenizer (Target Vocab Size: {target_vocab_size})...")
    tokenizer.train_from_iterator(get_training_corpus(dataset_dict), trainer=trainer)
    
    return MathTokenizer(tokenizer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_dir = config["paths"]["data_processed"]
    out_path = config["paths"]["tokenizer"]
    
    print(f"Loading processed datasets from {data_dir}...")
    try:
        dataset_dict = load_from_disk(data_dir)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_dir}. Run prepare_data.py first.")
        return

    target_vocab = config["model"].get("vocab_size", 16384)
    if config.get("smoke_test", False):
        target_vocab = min(target_vocab, 4000)

    math_tokenizer = build_and_train_tokenizer(dataset_dict, target_vocab_size=target_vocab)
    
    math_tokenizer.save(out_path)
    
    actual_vocab_size = math_tokenizer.vocab_size
    print(f"Tokenizer training complete! Actual Vocab Size: {actual_vocab_size}")
    print(f"Saved tokenizer to {out_path}")
    
    config["model"]["vocab_size"] = actual_vocab_size
    with open(args.config, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Updated config.yaml with vocab_size={actual_vocab_size}")

if __name__ == "__main__":
    main()

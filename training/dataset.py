import torch
from torch.utils.data import Dataset

class MathDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, objective_mode="reasoning", answer_loss_weight=5.0):
        self.max_seq_len = max_seq_len
        self.objective_mode = objective_mode
        self.answer_loss_weight = float(answer_loss_weight)

        r_id = tokenizer.r_id
        a_id = tokenizer.a_id
        pad_id = tokenizer.pad_id
        eos_id = tokenizer.eos_id

        print(f"Pre-tokenizing {len(dataset)} items for objective '{objective_mode}'...")
        
        prompts = []
        for item in dataset:
            q = item.get("question", "")
            r = item.get("reasoning", "")
            a = item.get("answer", "")

            if objective_mode == "direct" or not str(r).strip():
                prompts.append(f"[Q] {q} [A] {a} [EOS]")
            else:
                prompts.append(f"[Q] {q} [R] {r} [A] {a} [EOS]")

        encodings = tokenizer.tokenizer.encode_batch(prompts)

        all_input_ids = []
        all_labels = []
        all_weights = []

        for enc in encodings:
            ids = enc.ids
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                ids[-1] = eos_id

            lbls = list(ids)
            weights = [1.0] * len(ids)

            if objective_mode == "direct":
                try:
                    a_idx = ids.index(a_id)
                    for i in range(a_idx + 1):
                        lbls[i] = -100
                except ValueError:
                    pass
            else:  # "reasoning" or "weighted_reasoning"
                try:
                    r_idx = ids.index(r_id)
                    for i in range(r_idx + 1):
                        lbls[i] = -100
                except ValueError:
                    pass

                if objective_mode == "weighted_reasoning":
                    try:
                        a_idx = ids.index(a_id)
                        for i in range(a_idx, len(ids)):
                            if lbls[i] != -100:
                                weights[i] = self.answer_loss_weight
                    except ValueError:
                        pass

            seq_len = len(ids)
            if seq_len < max_seq_len:
                ids = ids + [pad_id] * (max_seq_len - seq_len)
                lbls = lbls + [-100] * (max_seq_len - seq_len)
                weights = weights + [0.0] * (max_seq_len - seq_len)

            all_input_ids.append(ids)
            all_labels.append(lbls)
            all_weights.append(weights)

        self.input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        self.loss_weights = torch.tensor(all_weights, dtype=torch.float32)
        print(f"Tensor dataset constructed: {self.input_ids.shape}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "loss_weights": self.loss_weights[idx]
        }

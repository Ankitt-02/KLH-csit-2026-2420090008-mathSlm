import torch
from torch.nn import functional as F
from collections import Counter
from verification.symbolic_verifier import SymbolicVerifier

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None, device="cpu", eos_id=None, repetition_penalty=1.0):
    """
    Autoregressive token generation using KV caching.
    """
    model.eval()
    input_ids = input_ids.to(device)
    past_key_values = None

    for _ in range(max_new_tokens):
        if past_key_values is not None:
            input_ids_curr = input_ids[:, -1:]
        else:
            input_ids_curr = input_ids

        logits, _, past_key_values = model(input_ids_curr, use_cache=True, past_key_values=past_key_values)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for b in range(input_ids.shape[0]):
                for token_id in set(input_ids[b].tolist()):
                    if logits[b, token_id] < 0:
                        logits[b, token_id] *= repetition_penalty
                    else:
                        logits[b, token_id] /= repetition_penalty

        if temperature == 0.0 or top_k == 1:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, idx_next), dim=1)

        if eos_id is not None and (idx_next == eos_id).all():
            break

    return input_ids

def parse_generated_text(text):
    """
    Extracts step-by-step reasoning and answer from output.
    Expected format: [Q] ... [R] reasoning [A] answer [EOS]
    """
    reasoning, answer = "", ""
    if "[R]" in text:
        parts = text.split("[R]")
        if len(parts) > 1:
            rest = parts[1]
            if "[A]" in rest:
                r_part, a_part = rest.split("[A]")
                reasoning = r_part.strip()
                answer = a_part.replace("[EOS]", "").strip()
            else:
                reasoning = rest.replace("[EOS]", "").strip()
    elif "[A]" in text:
        parts = text.split("[A]")
        if len(parts) > 1:
            answer = parts[1].replace("[EOS]", "").strip()

    if "\\boxed{" in answer:
        answer = answer.split("\\boxed{")[1].split("}")[0].strip()
    if " " in answer:
        answer = answer.split()[0]
    return reasoning, answer

def generate_self_consistency(model, tokenizer, prompt, num_samples=5, max_new_tokens=256, temperature=0.7, top_k=50, device="cpu"):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    samples = []
    answers = []

    for _ in range(num_samples):
        out_ids = generate(model, input_ids, max_new_tokens, temperature=temperature, top_k=top_k, device=device, eos_id=tokenizer.eos_id)
        out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

        reasoning, answer = parse_generated_text(out_text)
        samples.append({
            "full_text": out_text,
            "reasoning": reasoning,
            "answer": answer
        })
        if answer:
            answers.append(answer)

    if not answers:
        return samples[0], samples, {"vote_breakdown": {}}

    vote_counts = Counter(answers)
    majority_answer = vote_counts.most_common(1)[0][0]
    best_sample = next(s for s in samples if s["answer"] == majority_answer)

    return best_sample, samples, {"vote_breakdown": dict(vote_counts)}

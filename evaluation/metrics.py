import torch
from verification.symbolic_verifier import SymbolicVerifier

def calculate_exact_match(predicted, ground_truth):
    """
    Checks if predicted answer is exactly the ground truth (after stripping).
    For better evaluation, we use symbolic verifier equivalence instead of pure string match.
    """
    verifier = SymbolicVerifier()
    is_correct, _ = verifier.verify(reasoning="", stated_answer=predicted, ground_truth=ground_truth)
    return is_correct

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()

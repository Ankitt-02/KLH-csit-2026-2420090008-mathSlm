import re

class UntrainedSLM:
    """Mock for an untrained general purpose SLM for comparison."""
    def __init__(self):
        self.accuracy = 0.05 # Typically near 0 for math without fine tuning
        
    def predict(self, question):
        return "I don't know."

class RuleBasedSolver:
    """A simple regex-based solver for basic arithmetic."""
    def __init__(self):
        pass
        
    def predict(self, question):
        # Extremely naive: look for num op num
        match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', question)
        if match:
            n1, op, n2 = match.groups()
            n1, n2 = float(n1), float(n2)
            try:
                if op == '+': return str(n1 + n2)
                elif op == '-': return str(n1 - n2)
                elif op == '*': return str(n1 * n2)
                elif op == '/': return str(n1 / n2)
            except ZeroDivisionError:
                return ""
        return ""

class LargeLLMAPI:
    """
    Upper-bound baseline using a large LLM API.
    As agreed, we mock this with a public benchmark score to avoid API costs.
    E.g., GPT-4 zero-shot on GSM8K is ~92%.
    """
    def __init__(self):
        self.benchmark_score = 0.92
        
    def get_public_benchmark(self, dataset_name):
        if dataset_name.lower() == "gsm8k":
            return self.benchmark_score
        return 0.50 # arbitrary for others

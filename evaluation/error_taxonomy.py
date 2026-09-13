import re
import pandas as pd
from verification.symbolic_verifier import SymbolicVerifier

class ErrorTaxonomy:
    def __init__(self):
        self.verifier = SymbolicVerifier()

    def categorize(self, question, reasoning, predicted, ground_truth):
        """
        Heuristic-based error categorization.
        Categories:
        - arithmetic_error: The reasoning equation is correctly formulated, but calculated wrong.
        - unverifiable: Output is garbled or missing expected format.
        - wrong_operation: The model chose the wrong math operation.
        - hallucinated_step: The model added numbers not in the question.
        """
        if not predicted and not reasoning:
            return "unverifiable"
            
        if "[UNK]" in reasoning:
            return "unverifiable (contains unknown tokens)"
            
        # Check if numbers in reasoning exist in question
        nums_in_q = set(re.findall(r'\d+', question))
        nums_in_r = set(re.findall(r'\d+', reasoning))
        
        # If model is using numbers that aren't in the question (excluding small ints often used in math like 2, 100)
        suspicious_nums = nums_in_r - nums_in_q - {'1', '2', '10', '100'}
        if len(suspicious_nums) > 2:
            return "hallucinated_step"
            
        # Check internal arithmetic consistency
        lhs, rhs = self.verifier.extract_expression(reasoning)
        if lhs and rhs:
            is_consistent, _ = self.verifier.verify(reasoning)
            if not is_consistent:
                # If LHS != RHS in reasoning, it's an arithmetic error
                return "arithmetic_error"
            else:
                # If LHS == RHS but final answer is wrong, it means wrong operation was chosen
                return "wrong_operation"
                
        return "other_error"

def analyze_errors(results_csv):
    df = pd.read_csv(results_csv)
    incorrect_df = df[df['is_correct'] == False]
    
    taxonomy = ErrorTaxonomy()
    
    categories = []
    for _, row in incorrect_df.iterrows():
        cat = taxonomy.categorize(
            str(row['question']), 
            str(row['reasoning']), 
            str(row['predicted']), 
            str(row['ground_truth'])
        )
        categories.append(cat)
        
    incorrect_df['error_category'] = categories
    
    summary = incorrect_df['error_category'].value_counts()
    print("Error Taxonomy Summary:")
    print(summary)
    return summary

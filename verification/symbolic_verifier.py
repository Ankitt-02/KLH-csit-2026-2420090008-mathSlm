import re
import sympy

class SymbolicVerifier:
    def __init__(self):
        pass
        
    def extract_expression(self, text):
        """
        Extract the mathematical expression to be verified.
        In many cases, the final answer string from the model might just be a number,
        but if it provides an expression like "2 + 3", we extract it.
        For MathSLM, we assume the reasoning string can optionally be parsed, 
        but usually we just verify if the stated final answer is mathematically equivalent 
        to some ground truth, OR if the model outputs "Answer is 2+2=4", we check "2+2" == "4".
        
        A more robust approach for verification:
        Parse the final line of reasoning. If it contains an equation "LHS = RHS", verify it.
        """
        # Look for equations in the text
        equations = re.findall(r'([^=]+)=([^=]+)', text)
        if equations:
            # take the last equation
            lhs, rhs = equations[-1]
            return lhs.strip(), rhs.strip()
            
        return None, None

    def verify(self, reasoning, stated_answer=None, ground_truth=None):
        """
        1. If ground_truth is provided, just check if stated_answer is equivalent to it.
        2. If ground_truth is not provided, try to extract the last step of reasoning and verify it.
        """
        try:
            # Basic verification: check if stated answer matches ground truth mathematically
            if stated_answer and ground_truth:
                ans_expr = sympy.sympify(stated_answer.replace(',', ''))
                gt_expr = sympy.sympify(ground_truth.replace(',', ''))
                
                # Check equivalence
                is_correct = bool(sympy.simplify(ans_expr - gt_expr) == 0)
                return is_correct, ""
                
            # Internal consistency verification: check if LHS = RHS in the reasoning
            lhs, rhs = self.extract_expression(reasoning)
            if lhs and rhs:
                lhs_expr = sympy.sympify(lhs.replace(',', ''))
                rhs_expr = sympy.sympify(rhs.replace(',', ''))
                
                is_correct = bool(sympy.simplify(lhs_expr - rhs_expr) == 0)
                return is_correct, "Internal reasoning is mathematically sound." if is_correct else f"Mismatch: {lhs} != {rhs}"
                
            return False, "unverifiable"
            
        except Exception as e:
            return False, f"unverifiable ({str(e)})"

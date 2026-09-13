import os
import json
import random
import sympy
from datasets import Dataset, DatasetDict

def solve_ground_truth(expr_str):
    """
    Independent deterministic ground-truth solver using SymPy.
    Computes exact mathematical value for verification.
    """
    try:
        val = sympy.sympify(expr_str)
        return str(int(val))
    except Exception:
        return None

def generate_disjoint_dataset(num_train=8000, num_test=2000, seed=42):
    random.seed(seed)
    
    operations = ["addition", "subtraction", "multiplication", "division", "percentages", "algebra", "word_problems"]
    seen_problems = set()
    pool_by_op = {op: [] for op in operations}
    
    # 1. Addition
    for _ in range(3000):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        key = ("add", a, b)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"{a} + {b}")
        if gt_ans is None or gt_ans != str(a + b): continue
        
        q = f"What is {a} + {b}?"
        r = f"To add {a} and {b}, we combine the numbers: {a} + {b} = {gt_ans}."
        pool_by_op["addition"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "addition", "source": "synthetic"})

    # 2. Subtraction
    for _ in range(3000):
        a = random.randint(20, 999)
        b = random.randint(1, a)
        key = ("sub", a, b)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"{a} - {b}")
        if gt_ans is None or gt_ans != str(a - b): continue
        
        q = f"What is {a} - {b}?"
        r = f"To subtract {b} from {a}, we calculate: {a} - {b} = {gt_ans}."
        pool_by_op["subtraction"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "subtraction", "source": "synthetic"})

    # 3. Multiplication
    for _ in range(3000):
        a = random.randint(5, 99)
        b = random.randint(2, 50)
        key = ("mul", a, b)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"{a} * {b}")
        if gt_ans is None or gt_ans != str(a * b): continue
        
        q = f"What is {a} * {b}?"
        r = f"Multiplying {a} by {b} yields: {a} * {b} = {gt_ans}."
        pool_by_op["multiplication"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "multiplication", "source": "synthetic"})

    # 4. Division
    for _ in range(3000):
        b = random.randint(2, 25)
        ans_num = random.randint(2, 100)
        a = b * ans_num
        key = ("div", a, b)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"{a} / {b}")
        if gt_ans is None or gt_ans != str(ans_num): continue
        
        q = f"What is {a} / {b}?"
        r = f"Dividing {a} by {b} gives: {a} / {b} = {gt_ans}."
        pool_by_op["division"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "division", "source": "synthetic"})

    # 5. Percentages
    for _ in range(3000):
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        base = random.randint(1, 50) * 20
        ans_num = int((pct / 100.0) * base)
        key = ("pct", pct, base)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"({pct} / 100) * {base}")
        if gt_ans is None or gt_ans != str(ans_num): continue
        
        q = f"What is {pct}% of {base}?"
        r = f"To find {pct}% of {base}, we calculate ({pct}/100) * {base} = {gt_ans}."
        pool_by_op["percentages"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "percentages", "source": "synthetic"})

    # 6. Simple Algebra
    for _ in range(3000):
        a = random.randint(2, 12)
        x = random.randint(1, 30)
        b = random.randint(1, 50)
        c = a * x + b
        key = ("alg", a, b, c)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"({c} - {b}) / {a}")
        if gt_ans is None or gt_ans != str(x): continue
        
        q = f"Solve {a}x + {b} = {c}."
        r = f"Subtract {b} from both sides: {a}x = {c-b}. Divide by {a}: x = {gt_ans}."
        pool_by_op["algebra"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "algebra", "source": "synthetic"})

    # 7. One-step word problems
    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry"]
    items = ["apples", "books", "marbles", "pencils", "stickers", "cards"]
    for _ in range(3000):
        name = random.choice(names)
        item = random.choice(items)
        n1 = random.randint(5, 50)
        n2 = random.randint(5, 50)
        key = ("word", name, item, n1, n2)
        if key in seen_problems: continue
        seen_problems.add(key)
        
        gt_ans = solve_ground_truth(f"{n1} + {n2}")
        if gt_ans is None or gt_ans != str(n1 + n2): continue
        
        q = f"{name} has {n1} {item}. They bought {n2} more. How many {item} do they have in total?"
        r = f"{name} started with {n1} and added {n2}: {n1} + {n2} = {gt_ans}."
        pool_by_op["word_problems"].append({"question": q, "reasoning": r, "answer": gt_ans, "subject": "word_problems", "source": "synthetic"})

    train_samples = []
    test_samples = []
    
    # Guarantee strict disjoint numerical pairs between train and test
    for op, samples in pool_by_op.items():
        random.shuffle(samples)
        split_idx = int(0.8 * len(samples))
        train_samples.extend(samples[:split_idx])
        test_samples.extend(samples[split_idx:])

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    train_samples = train_samples[:num_train]
    test_samples = test_samples[:num_test]

    print(f"Generated and SymPy-validated {len(train_samples)} disjoint train samples and {len(test_samples)} held-out test samples across {len(operations)} operations.")

    ds_train = Dataset.from_list(train_samples)
    ds_test = Dataset.from_list(test_samples)

    return DatasetDict({"train": ds_train, "val": ds_test, "test": ds_test})

if __name__ == "__main__":
    ds_dict = generate_disjoint_dataset(num_train=8000, num_test=2000)
    out_dir = "data/processed_synthetic"
    ds_dict.save_to_disk(out_dir)
    print(f"Validated synthetic dataset saved to {out_dir}")

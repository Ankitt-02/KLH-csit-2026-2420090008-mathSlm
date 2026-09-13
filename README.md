# MathSLM: A Transformer-Based Small Language Model for Mathematical Reasoning

MathSLM is a lightweight, decoder-only Transformer Small Language Model (SLM) trained from scratch to solve mathematical word problems with step-by-step reasoning. It includes a **symbolic execution layer** using SymPy for answer verification and is served through a **Streamlit web app**.

## Project Thesis
You don't need a massive LLM to get accurate, explainable math reasoning. A small, efficient, purpose-built Transformer combined with a symbolic verifier and self-consistency decoding can close most of the gap to large models while running on modest hardware.

## Quick Start (Smoke Test & Diagnostics)

We provide both a local CPU smoke test configuration and Google Colab GPU diagnostic pipeline.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data (Tiny Subset)**
   ```bash
   python data/prepare_data.py
   ```

3. **Train Tokenizer**
   ```bash
   python tokenizer/train_tokenizer.py
   ```

4. **Train Model (CPU / CUDA)**
   ```bash
   python training/train.py
   ```

5. **Run Google Colab Diagnostics (V2 Matrix)**
   Open `notebooks/MathSLM_v2_diagnostics.ipynb` in Google Colab to run GPU environment verification, synthetic disjoint benchmark generation, Tiny Sanity Test, and the 4-way controlled experiment matrix (Exp-A through Exp-D).

6. **Run the App**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Architecture
- **Tokenizer**: Custom BPE with digit-aware tokenization to prevent subword scrambling of arithmetic.
- **Model**: Decoder-only Transformer built from scratch in PyTorch with AMP mixed precision and hardware-agnostic CUDA/CPU device routing.
- **Verification**: SymPy-based symbolic verification to parse output equations and mathematically check equivalence.
- **Colab Diagnostics**: 4-way controlled matrix isolating undertraining, model capacity limits, and reasoning loss dilution.

## Team Members

| S.No. | Roll Number / ID | Team Member  |
| ----- | ---------------- | ------------ |
| 1     | 2420090050       | Rupen Parthu |
| 2     | 2420090008       | Ankit Swami  |
| 3     | 2420030635       | Nikhil Sai   |

**Supervisor:** Dr. K Swanthana

# MathSLM: A Transformer-Based Small Language Model for Mathematical Reasoning.

## 1. Project Overview

**MathSLM** is a Transformer-based Small Language Model designed for mathematical reasoning. The project aims to develop a lightweight and computationally efficient language model capable of understanding mathematical problems and generating step-by-step solutions.

Large Language Models can perform mathematical problem solving effectively, but their computational requirements and deployment costs can make them unsuitable for resource-constrained educational environments. MathSLM addresses this challenge by developing a smaller, domain-focused language model for mathematical reasoning.

The proposed system uses a **decoder-only Transformer architecture** implemented using the **PyTorch** deep learning framework. The model is trained on publicly available mathematical reasoning datasets and learns mathematical patterns through next-token prediction. It is designed to generate intermediate reasoning steps before producing the final mathematical answer.

---

## 2. Project Objectives

The main objectives of MathSLM are:

* Develop a lightweight Transformer-based Small Language Model for mathematical reasoning.
* Process and prepare mathematical reasoning datasets for model training.
* Perform mathematical text preprocessing and tokenization.
* Train a decoder-only Transformer using next-token prediction.
* Generate step-by-step mathematical reasoning.
* Support configurable text generation strategies.
* Evaluate the model using quantitative and qualitative metrics.
* Provide a web-based interface for interacting with the trained model.

---

## 3. Team Members

| S.No. | Roll Number / ID | Team Member  |
| ----- | ---------------- | ------------ |
| 1     | 2420090050       | Rupen Parthu |
| 2     | 2420090008       | Ankit Swami  |
| 3     | 2420030635       | Nikhil Sai   |

The team member information is taken from the project abstract.

### Supervisor

**Supervisor:** Dr. K Swanthana

---

## 4. Abstract

Mathematical reasoning is a fundamental capability required in educational systems, scientific computing, and intelligent tutoring applications. While large language models have demonstrated strong performance on mathematical problem solving, their high computational requirements and deployment costs make them unsuitable for many educational and resource-constrained environments.

MathSLM proposes a lightweight Transformer-based framework designed to understand, reason, and generate step-by-step solutions for mathematical problems while maintaining computational efficiency.

The proposed system implements a decoder-only Transformer architecture using the PyTorch deep learning framework and trains it on publicly available mathematical reasoning datasets. The framework performs data preprocessing, tokenization, sequence preparation, Transformer-based language modeling, supervised training, and autoregressive text generation.

The model learns mathematical patterns through next-token prediction and generates intermediate reasoning steps before producing the final answer. Configurable generation strategies including greedy decoding, temperature sampling, and top-k sampling are incorporated during inference.

A web-based interface enables users to enter mathematical queries and interact with the trained model in real time. The system is evaluated using training and validation loss, perplexity, token prediction accuracy, benchmark evaluation, and qualitative analysis of generated solutions.

---

## 5. System Architecture

The MathSLM workflow consists of the following stages:

```text
Mathematical Reasoning Datasets
            |
            v
     Data Preprocessing
            |
            v
     Mathematical Tokenization
     (Hugging Face Tokenizer / BPE)
            |
            v
      Sequence Preparation
   (Input IDs, Attention Masks,
          Labels)
            |
            v
   Decoder-Only Transformer
   - Embedding
   - Multi-Head Attention
   - Feed Forward Network
            |
            v
      Language Model Head
            |
            v
     Next-Token Prediction
            |
            v
      Trained MathSLM
            |
            v
  +---------+----------+
  |         |          |
Greedy    Top-k     Temperature
Decode   Sampling    Sampling
  |         |          |
  +---------+----------+
            |
            v
Step-by-Step Mathematical
Reasoning Generation
            |
            v
   Mathematical Solution
            |
            v
    Streamlit Web Application
```

The architecture and processing flow are based on the approach diagram provided in the project abstract.

---

## 6. Technology Stack

### Programming Language

* Python

### Deep Learning Framework

* PyTorch

### Model Architecture

* Decoder-only Transformer
* Embedding layer
* Multi-head attention
* Feed-forward network
* Language model head
* Next-token prediction

### Tokenization

* Hugging Face tokenizer
* Byte Pair Encoding (BPE)

### Web Interface

* Streamlit

### Dataset / Data Processing

* Mathematical reasoning datasets
* Data preprocessing
* Sequence preparation
* Input IDs
* Attention masks
* Labels

The project abstract specifically identifies PyTorch, decoder-only Transformer architecture, Hugging Face tokenizer/BPE, and Streamlit as components of the proposed system.

---

## 7. Datasets

The project documentation identifies the following mathematical reasoning datasets:

| Dataset      | Purpose / Description                               | Source                                                     |
| ------------ | --------------------------------------------------- | ---------------------------------------------------------- |
| GSM8K        | Grade School Math 8K mathematical reasoning dataset | https://huggingface.co/datasets/gsm8k                      |
| MathQA       | Mathematical question-answering dataset             | https://huggingface.co/datasets/math_qa                    |
| MATH Dataset | Mathematical competition/reasoning dataset          | https://huggingface.co/datasets/hendrycks/competition_math |
| NuminaMath   | Optional mathematical reasoning dataset             | https://huggingface.co/datasets/AI-MO/NuminaMath           |

These dataset sources are listed in the project abstract.

### Datasets Currently Present in This Repository

The repository currently contains:

* GSM8K
* Hendrycks MATH

The `/data` directory contains the available dataset files.

> **Note:** Only datasets that are legally permitted to be redistributed should be committed to this repository. If a dataset cannot be redistributed, the `/data` directory should instead contain documented source and download instructions.

---

## 8. Data Processing Pipeline

The data processing pipeline consists of:

1. Dataset collection
2. Data preprocessing
3. Mathematical text tokenization
4. Sequence preparation
5. Creation of input IDs
6. Creation of attention masks
7. Creation of training labels
8. Preparation of data for Transformer-based language modeling

The model then uses the prepared sequences for supervised next-token prediction.

---

## 9. Model

MathSLM uses a **decoder-only Transformer** architecture.

The model contains:

* Token embeddings
* Positional information
* Multi-head self-attention
* Feed-forward layers
* Transformer blocks
* Language model head

The model is trained using **next-token prediction**, allowing it to learn mathematical language patterns and generate solutions autoregressively.

---

## 10. Text Generation

MathSLM supports configurable text generation strategies:

### Greedy Decoding

Selects the most probable next token at every generation step.

### Temperature Sampling

Controls the randomness of token selection during generation.

### Top-k Sampling

Restricts token selection to the top-k most probable candidate tokens.

These strategies are used to control the quality and diversity of generated mathematical reasoning.

---

## 11. Evaluation

The model is evaluated using both quantitative and qualitative methods.

### Quantitative Evaluation

The planned evaluation metrics include:

* Training loss
* Validation loss
* Perplexity
* Token prediction accuracy
* Benchmark evaluation on mathematical reasoning datasets

### Qualitative Evaluation

Generated responses are assessed based on:

* Correctness
* Coherence
* Quality of mathematical reasoning
* Step-by-step reasoning
* Final answer quality

The evaluation approach is described in the project abstract.

---

## 12. Web Application

MathSLM includes a web-based interface through which users can enter mathematical questions and interact with the trained model in real time.

The intended workflow is:

```text
User enters mathematical problem
              |
              v
       Streamlit Interface
              |
              v
        Trained MathSLM
              |
              v
   Autoregressive Generation
              |
              v
 Step-by-Step Mathematical Solution
```

The project abstract identifies Streamlit as the web-based interface for interacting with the trained model.

---

## 13. Repository Structure

```text
KLH-cse-2026-2420030635-mathslm/
│
├── data/
│   ├── gsm8k/
│   ├── hendrycks_math/
│   └── README.md
│
├── docs/
│   ├── ALT_ABSTRACT.pdf
│   └── project documentation
│
├── notebooks/
│   └── experimentation notebooks
│
├── reports/
│   ├── review-1/
│   ├── review-2/
│   └── final/
│
├── results/
│   ├── evaluation/
│   ├── graphs/
│   └── predictions/
│
├── src/
│   ├── preprocessing/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── tests/
│   └── project tests
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 14. Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/NIKHILSAI4/KLH-cse-2026-2420030635-mathslm.git
cd KLH-cse-2026-2420030635-mathslm
```

### Step 2: Create a Python Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Prepare the Dataset

Place the required datasets under:

```text
data/
```

Refer to `data/README.md` for dataset source and preparation instructions.

---

## 15. Execution

The project consists of the following general execution stages:

```text
Dataset
   ↓
Preprocessing
   ↓
Tokenization
   ↓
Sequence Preparation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Text Generation
   ↓
Streamlit Application
```

Run the corresponding scripts from the `src/` directory after the implementation is finalized.

> **Execution commands will be updated here as the implementation is finalized.**

---

## 16. Current Phase Status

### Current Phase: `[ENTER CURRENT PHASE / REVIEW NUMBER]`

**Status:** `[In Progress / Completed]`

### Completed

* [x] Project repository created
* [x] Mandatory repository folder structure created
* [x] Mathematical reasoning datasets prepared
* [x] Project abstract and architecture documented

### In Progress

* [ ] Data preprocessing pipeline
* [ ] Mathematical tokenization
* [ ] Sequence preparation
* [ ] Decoder-only Transformer implementation
* [ ] Model training
* [ ] Model evaluation
* [ ] Text generation
* [ ] Streamlit web application
* [ ] Final documentation

### Upcoming

* [ ] Complete model training
* [ ] Evaluate model on mathematical reasoning datasets
* [ ] Analyze generated solutions
* [ ] Complete web application
* [ ] Prepare final project report
* [ ] Final evaluation and demonstration

> This section will be updated after every project phase/review.

---

## 17. Project Phase Tags

Project phase deliverables are maintained using Git tags.

Planned tags:

```text
review-1
review-2
final
```

Each tag represents the repository state corresponding to the respective project phase.

---

## 18. Contribution Guidelines

All team members contribute using their own GitHub accounts so that individual contributions remain verifiable.

Team members:

* Rupen Parthu — 2420090050
* Ankit Swami — 2420090008
* Nikhil Sai — 2420030635

Meaningful commits are maintained progressively throughout the project.

---

## 19. Security and Data Policy

The repository must not contain:

* Passwords
* API keys
* Authentication tokens
* Private credentials
* Confidential institutional information
* Unauthorized licensed datasets

Sensitive information must be stored outside the repository and must never be committed.

The `.gitignore` file is used to prevent accidental submission of sensitive or unnecessary local files.

---

## 20. License and Dataset Notice

Dataset ownership and licensing remain with their respective dataset providers.

Before redistributing any dataset through this repository, its license and redistribution permissions must be verified.

Where redistribution is not permitted, the repository will provide the dataset source and instructions for obtaining the dataset instead of storing the dataset files.

---

## 21. Future Scope

The project can serve as a foundation for future work in:

* Domain-specific Small Language Models
* Mathematical intelligence
* Educational AI
* Mathematical tutoring systems
* Efficient language model deployment
* Improved mathematical reasoning
* Further dataset expansion
* More advanced evaluation methods

The project abstract identifies domain-specific language models, mathematical intelligence, and educational AI applications as potential future directions.

---

## 22. Team

**Project:** MathSLM: A Transformer-Based Small Language Model for Mathematical Reasoning

**Team Members:**

1. **Rupen Parthu** — 2420090050
2. **Ankit Swami** — 2420090008
3. **Nikhil Sai** — 2420030635


**Repository:** `KLH-cse-2026-2420030635-mathslm`

#!/bin/bash
set -e

echo "============================================================"
echo "MathSLM v2 - Google Colab Environment Setup"
echo "============================================================"

# Install required Python packages
echo "Installing dependencies..."
pip install --quiet torch tokenizers datasets pandas pyyaml tqdm sympy

# Check PyTorch CUDA availability
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    print('GPU Memory (Total):', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('WARNING: CUDA is not available. Execution will fallback to CPU.')
"

# Create required directories
mkdir -p data checkpoints logs notebooks

echo "Environment setup complete!"

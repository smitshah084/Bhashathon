# Install PyTorch with CUDA 12.0 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia

# Install DeepSpeed (requires compilation)
pip install deepspeed

# Hugging Face and training dependencies
pip install \
    transformers \
    datasets \
    tokenizers \
    tqdm \
    tensorboardX \
    wandb \
    scikit-learn \
    pandas \
    matplotlib \
    nvidia-ml-py

# Verify installations
python -c "
import torch
import deepspeed
print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'DeepSpeed version: {deepspeed.__version__}')
"

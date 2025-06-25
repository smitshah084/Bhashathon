# Bhashathon: Training and Sampling Instructions

This repository contains the implementation of a GPT-2 based language model trained from scratch for next-token prediction using unigram tokenization. The model is optimized for Indic languages.

## Setup AWS Instance G4dn or G5

### 1. Update and Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl wget git python3-pip
```

### 2. Install NVIDIA Drivers and CUDA Toolkit
```bash
sudo apt install -y nvidia-driver-535
sudo reboot
nvidia-smi

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update && sudo apt upgrade
sudo apt install -y cuda-toolkit
```

### 3. Mount Storage Drive
```bash
lsblk
sudo fdisk -l /dev/nvme1n1
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data
df -h
sudo chown -R $(whoami):$(whoami) /mnt/
sudo nano /etc/fstab
```
Add the following line to `/etc/fstab` to ensure persistence across reboots:
```
/dev/nvme1n1  /data  ext4  defaults,nofail  0  2
```

## Setup Conda Environment
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /mnt/data/miniconda-installer.sh
bash /data/miniconda-installer.sh
echo 'export PATH="/data/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init
```

### Clone Repository and Setup Environment
```bash
git clone https://github.com/smitshah084/Bhashathon.git
cd Bhashathon
conda env create -f aws_env.yml
conda activate aws_env
pip install gdown sentencepiece 
```

## Download txt Data

### Processed Bin Data V1: 666M tokens
```bash
gdown "https://drive.google.com/uc?export=download&id=1qU2cyImtr2anDXefYvxaPw1GD6tOojng"
gdown --id 15qZ97WGFwdojRyEKQfWCOnBagaquUJy5
gdown --id 1AL9qa1qE98F7a2VECS7sL4fojp9KV5es
gdown --id 1I7WlvtsIIDGahIBxZ0qKpkSGPUqVg4fI
gdown --id 1_RAxJnzgz4EaQwNY7p3G04_zUq2YUAsF
gdown --id 11ne763lj58m9b4ZCWEUGKBPb-a2CW8jN
gdown --id 19UuTyxvnVhp0jMl-55sgpNR1jpeeX-D3
```

### CFILT text Dataset
```bash
wget https://www.cfilt.iitb.ac.in/eilmt_challenge_round_dataserver/Word_prediction_bpcc_dataset/train.guj_Gujr
wget https://www.cfilt.iitb.ac.in/eilmt_challenge_round_dataserver/Word_prediction_bpcc_dataset/train.hin_Deva
wget https://www.cfilt.iitb.ac.in/eilmt_challenge_round_dataserver/Word_prediction_bpcc_dataset/train.kan_Knda
wget https://www.cfilt.iitb.ac.in/eilmt_challenge_round_dataserver/Word_prediction_bpcc_dataset/train.mal_Mlym
wget https://www.cfilt.iitb.ac.in/eilmt_challenge_round_dataserver/Word_prediction_bpcc_dataset/train.mar_Deva
wget https://www.cfilt.iitb.ac.in/eilmt_challenge_round_dataserver/Word_prediction_bpcc_dataset/train.ory_Orya
```

### Large Indic Corpora text data set
```bash
wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt
wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/mr.txt
wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/ml.txt
wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/kn.txt
wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/gu.txt
```

### Processed Bin Data V2: 3.2 Billion tokens
```bash
gdown --id 16GNK3pRnx8V-WFa_8_lk5Q9qn6sLXqfQ
gdown --id 1MrxhijpGw8aDZ4ecy0ct7VKfvBpGkeug
gdown --id 1rvvF6jKxK1PZrRwk2X_12bD_p1DOc3-2
gdown --id 1os-BZ68pcd_cTYSoYQu8r2JxaHiZwRMT
gdown --id 1SlP7-BCWohIwAGTtYbHaPVMUvzXBM4lR
gdown --id 1CE0OFV5G-DcfzYROdoc_dqOGyzY8nZZm
gdown --id 10SacUhzkuwjd2GzCMershMpF0-N8QMK4
gdown --id 1OD9YMz8Oeghf_ARQ2Mxd97Caph_0LZK2
gdown --id 11PgoGsOp4mswUIULqN69lmOVnXd6JTO6
gdown --id 1HigFC1f00mYxlkhYBGTLGRLqUXncE8RF
gdown --id 1vjkNwVCJARvkwfIoJCpfu476mJPgVvNP
gdown --id 1AuPBblarT2U8RKU8DgNVogVj49K_ynx2
gdown --id 1GpBGx2BY5ijou9kAK7sb4Qw8fZFCNnlX
```

## Training and Sampling

### Train Model
```bash
python train.py
```

### Download Model Checkpoint : 8200 iterations
``` bash
gdown 1V0LBSF9KOdunFLXXc4cfIu6AfkmHmu-1
```

### Sample from Model
```bash
python sample.py
```

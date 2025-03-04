from Preprocessing.data_loader import FastBinaryDataLoader

train_bin_path = 'bin_data/train.bin'
test_bin_paths = ['./bin_data/Hindi_test.bin',
 './bin_data/gujrati_test.bin',
 './bin_data/kannada_test.bin',
 './bin_data/mar_test.bin',
 './bin_data/mlym_test.bin',
 './bin_data/orya_test.bin',]

langs = ["Hin","Guj","Kann","Mar","mlym","Orya"]

for path in test_bin_paths:
        
        print(path,end=" ")

        test_loader = FastBinaryDataLoader(
        data_path=path, 
        batch_size=64, 
        shuffle=False,  # No need to shuffle test data
        sequence_length=1024, 
        buffer_size=250_000, 
        prefetch=False  # No need to prefetch in test
        )
        
        num_test_batches = len(test_loader)
        print(":",num_test_batches)


import os
import time
import torch
import argparse
import numpy as np
import deepspeed
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from Preprocessing.data_loader import FastBinaryDataLoader

# Set up argument parser
parser = argparse.ArgumentParser(description='Train GPT-2 1.5B model')
parser.add_argument('--train_path', type=str, default='./bin_data/train.bin', help='Path to training data')
parser.add_argument('--test_path', type=str, default='./test.bin', help='Path to validation data')
parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
args = parser.parse_args()

# Create output and log directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize tensorboard
writer = SummaryWriter(log_dir=args.log_dir)

# GPT-2 1.5B model configuration (standard hyperparameters as published)
model_config = GPT2Config(
    vocab_size=50304,
    n_positions=1024,
    n_embd=1600,
    n_layer=48,
    n_head=25,
    activation_function='gelu_new',
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    summary_type='cls_index',
    summary_use_proj=True,
    summary_activation=None,
    summary_proj_to_labels=True,
    summary_first_dropout=0.1
)

# DeepSpeed configuration optimized for your hardware
ds_config = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "steps_per_print": 10,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": 48
    },
    "wall_clock_breakdown": True,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 6e-5,
            "warmup_num_steps": 3000
        }
    }
}

# Initialize model
model = GPT2LMHeadModel(model_config)

# Setup data loaders
# Parameters optimized for your hardware
sequence_length = 1024
train_batch_size = ds_config["train_micro_batch_size_per_gpu"]
buffer_size = 10000  # Buffer to efficiently load data
prefetch = 4  # Number of batches to prefetch

train_dataloader = FastBinaryDataLoader(
    data_path=args.train_path,
    batch_size=train_batch_size,
    shuffle=True,
    sequence_length=sequence_length,
    buffer_size=buffer_size,
    prefetch=prefetch
)

test_dataloaders = [FastBinaryDataLoader(
    data_path=path,
    batch_size=train_batch_size,
    shuffle=False,
    sequence_length=sequence_length,
    buffer_size=buffer_size,
    prefetch=prefetch
) for path in test_bin_paths]

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    dist_init_required=True
)

# Training settings
epochs = 5
gradient_accumulation_steps = ds_config["gradient_accumulation_steps"]
steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
total_steps = steps_per_epoch * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=3000,
    num_training_steps=total_steps
)

# Training monitoring
best_val_loss = float('inf')
start_time = time.time()

def log_memory_usage():
    """Log GPU and RAM memory usage"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
        stats = {
            'gpu_memory_allocated_gb': gpu_memory_allocated,
            'gpu_memory_reserved_gb': gpu_memory_reserved,
        }
        return stats
    return {}

def evaluate(writer=None):
    """Evaluate the model on the test set"""
    model_engine.eval()
    avg_loss = 0
    perplexity = 0
    for id,loader in enumerate(test_dataloaders):
        total_loss = 0
        with torch.no_grad():
                for step, batch in enumerate(tqdm(loader, desc="Evaluating")):
                        batch = batch.to(model_engine.device)
                        outputs = model_engine(batch, labels=batch)
                        loss = outputs.loss
                        total_loss += loss.item()

        # Calculate perplexity
        avg_loss_1 = total_loss / len(loader)
        perplexity_1 = torch.exp(torch.tensor(avg_loss)).item()
        if writer:
                writer.add_scalar(langs[id]+"_loss",avg_loss_1)
                writer.add_scalar(langs[id]+"_perplexity",perplexity_1)
        avg_loss += avg_loss_1/6
        perplexity += perplexity_1/6
        
    return avg_loss, perplexity

# Training loop
global_step = 0
log_interval = 10
eval_interval = 500

print(f"Starting training with {epochs} epochs, {steps_per_epoch} steps per epoch")
for epoch in range(epochs):
    model_engine.train()
    epoch_start_time = time.time()
    running_loss = 0
    
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
        batch = batch.to(model_engine.device)
        outputs = model_engine(batch, labels=batch)
        loss = outputs.loss
        model_engine.backward(loss)
        
        # Update weights after gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            model_engine.step()
            running_loss += loss.item()
            global_step += 1
            
            # Log training stats
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                tokens_per_sec = (train_batch_size * sequence_length * log_interval) / (time.time() - start_time)
                memory_stats = log_memory_usage()
                
                print(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {avg_loss:.4f}, "
                      f"Tokens/sec: {tokens_per_sec:.2f}, "
                      f"GPU Memory: {memory_stats.get('gpu_memory_allocated_gb', 0):.2f} GB")
                
                writer.add_scalar('train/loss', avg_loss, global_step)
                writer.add_scalar('train/tokens_per_sec', tokens_per_sec, global_step)
                for key, value in memory_stats.items():
                    writer.add_scalar(f'resources/{key}', value, global_step)
                
                running_loss = 0
                start_time = time.time()
            
            # Evaluate model
            if global_step % eval_interval == 0:
                val_loss, perplexity = evaluate()
                print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
                writer.add_scalar('eval/loss', val_loss, global_step)
                writer.add_scalar('eval/perplexity', perplexity, global_step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_engine.save_checkpoint(args.output_dir, tag=f"best_model_{global_step}")
                    print(f"New best model saved at step {global_step}")
                
                model_engine.train()
    
    # End of epoch
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
    
    # Save checkpoint at end of each epoch
    model_engine.save_checkpoint(args.output_dir, tag=f"epoch_{epoch+1}")
    
    # Evaluate at the end of epoch
    val_loss, perplexity = evaluate()
    print(f"End of Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
    writer.add_scalar('eval/epoch_loss', val_loss, epoch+1)
    writer.add_scalar('eval/epoch_perplexity', perplexity, epoch+1)

# Final evaluation
final_val_loss, final_perplexity = evaluate()
print(f"Training completed - Final Validation Loss: {final_val_loss:.4f}, Perplexity: {final_perplexity:.2f}")

# Save final model
model_engine.save_checkpoint(args.output_dir, tag="final_model")

# Close tensorboard writer
writer.close()
print(f"Model saved to {args.output_dir}")

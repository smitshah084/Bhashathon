import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm 
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from collections import deque
from model import GPTConfig, GPT
import glob

out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 3
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
# 0dcd4aa8bbcc0cee524f0ce847250dafda4f1024
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset_dir = 'bin_data'
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
gradient_accumulation_steps = int(500000/(batch_size*1024))# used to simulate larger batch sizes
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
epoch = 1
max_iters = (8981//batch_size) * epoch # total number of training iterations

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

metrics = {
    'train_tokens_per_second': deque(maxlen=100),  # Forward pass speed
    'test_tokens_per_second': deque(maxlen=100),   # Evaluation speed
    'param_update_time': deque(maxlen=20),         # Time for parameter updates
    'data_load_time': deque(maxlen=100),           # Time to load batches
    'batch_processing_time': {                     # Time to process batches (train/test)
        'train': deque(maxlen=100),
        'test': deque(maxlen=100)
    }
}


ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def get_batch(split, batch=None, language=None):
    data_load_start = time.time()
    B = batch_size if batch is None else batch
    
    # Get all available files for the requested split
    file_pattern = f"*_{split}.bin"
    all_files = glob.glob(os.path.join(dataset_dir, file_pattern))
    
    # If language is specified, filter files for that language
    if language:
        all_files = [f for f in all_files if language in os.path.basename(f)]
    
    if not all_files:
        raise ValueError(f"No {split} data files found for {'specified language' if language else 'any language'}")
    
    # Get file sizes to determine sampling ratio
    file_sizes = []
    file_data = []
    
    for file_path in all_files:
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
        file_sizes.append(len(data))
        file_data.append(data)
    
    total_tokens = sum(file_sizes)
    sampling_probs = [size / total_tokens for size in file_sizes]
    
    # Sample batch_size sequences proportional to file sizes
    x_list = []
    y_list = []
    
    # Count how many sequences to sample from each file
    samples_per_file = [int(B * prob) for prob in sampling_probs]
    # Ensure we get exactly B samples by adding any remainder to the largest file
    remainder = B - sum(samples_per_file)
    if remainder > 0:
        max_idx = sampling_probs.index(max(sampling_probs))
        samples_per_file[max_idx] += remainder
    
    # Sample from each file
    for i, (data, num_samples) in enumerate(zip(file_data, samples_per_file)):
        if num_samples == 0:
            continue
            
        # Only sample if there's enough data
        if len(data) <= block_size:
            print(f"Warning: File {all_files[i]} is too small, skipping")
            continue
            
        # Random indices
        ix = torch.randint(len(data) - block_size, (num_samples,))
        
        # Get x and y sequences
        for idx in ix:
            x = torch.from_numpy((data[idx:idx+block_size]).astype(np.int64))
            y = torch.from_numpy((data[idx+1:idx+1+block_size]).astype(np.int64))
            x_list.append(x)
            y_list.append(y)
    
    # Stack all samples
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    # Move to device
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    data_load_end = time.time()
    metrics['data_load_time'].append(data_load_end - data_load_start)
    
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

meta_vocab_size = 50304

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    split = 'test'
    for lang in ['gujrati','hindi','kannada','malyalam','marathi','odia']:
        losses = torch.zeros(eval_iters)
        tokens_processed = 0
        test_start_time = time.time()
        print("ON LANG: ", lang)
        
        for k in range(eval_iters):
            print("K: ", k)
            batch_start = time.time()
            test_batch_size = int(batch_size * 1.5)
            X1, Y1 = get_batch(split, test_batch_size,lang)
            tokens_processed += test_batch_size * block_size
            
            with ctx:
                _, loss = model(X1, Y1)  # Fixed X, Y to X1, Y1
            
            losses[k] = loss.item()
            batch_end = time.time()
            metrics['batch_processing_time']['test'].append(batch_end - batch_start)
            
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        
        # Calculate tokens per second during test
        if test_duration > 0:
            test_tokens_per_second = tokens_processed / test_duration
            metrics['test_tokens_per_second'].append(test_tokens_per_second)
            print(f"{lang} tokens/sec: {test_tokens_per_second:.2f}")
            
        out[lang] = losses.mean()
    
    out['val'] = sum(out.values()) / len(out)
    model.train()
    return out

def print_performance_metrics(iter_num, total_tokens_processed):
    avg_train_tokens_per_second = np.mean(metrics['train_tokens_per_second']) if metrics['train_tokens_per_second'] else 0
    avg_test_tokens_per_second = np.mean(metrics['test_tokens_per_second']) if metrics['test_tokens_per_second'] else 0
    avg_param_update_time = np.mean(metrics['param_update_time']) if metrics['param_update_time'] else 0
    avg_data_load_time = np.mean(metrics['data_load_time']) if metrics['data_load_time'] else 0
    avg_train_batch_time = np.mean(metrics['batch_processing_time']['train']) if metrics['batch_processing_time']['train'] else 0
    avg_test_batch_time = np.mean(metrics['batch_processing_time']['test']) if metrics['batch_processing_time']['test'] else 0
    
    # Estimate time to train on 9.1M tokens
    remaining_tokens = 9_100_000 - total_tokens_processed
    estimated_time_seconds = remaining_tokens / avg_train_tokens_per_second if avg_train_tokens_per_second > 0 else 0
    hours, remainder = divmod(estimated_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"PERFORMANCE METRICS (iter {iter_num}):")
    print(f"Training tokens/second: {avg_train_tokens_per_second:.2f}")
    print(f"Testing tokens/second: {avg_test_tokens_per_second:.2f}")
    print(f"Parameter update time: {avg_param_update_time*1000:.2f}ms (frequency: every {gradient_accumulation_steps} micro steps)")
    print(f"Data loading time: {avg_data_load_time*1000:.2f}ms per batch")
    print(f"Train batch processing time: {avg_train_batch_time*1000:.2f}ms")
    print(f"Test batch processing time: {avg_test_batch_time*1000:.2f}ms")
    print(f"Estimated time to train on 9.1M tokens: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*50 + "\n")


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

total_tokens_processed = 0

# First batch fetch - add timing
data_load_start = time.time()
X, Y = get_batch('train')
data_load_end = time.time()
metrics['data_load_time'].append(data_load_end - data_load_start)

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

try:
    for iter_num in tqdm(range(max_iters), desc="Training Progress", dynamic_ncols=True):
        print("ITER: ",iter_num)
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process and iter_num != 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Add metrics to wandb logging
            if wandb_log:
                avg_train_tokens_per_second = np.mean(metrics['train_tokens_per_second']) if metrics['train_tokens_per_second'] else 0
                avg_test_tokens_per_second = np.mean(metrics['test_tokens_per_second']) if metrics['test_tokens_per_second'] else 0
                avg_param_update_time = np.mean(metrics['param_update_time']) if metrics['param_update_time'] else 0
                
                wandb.log({
                    "iter": iter_num,
                    "gujrati/loss": losses['gujrati'],
                    "hindi/loss": losses['hindi'],
                    "Hindi_test/loss": losses['Hindi_test'],
                    "kannada/loss": losses['kannada'],
                    "malyalam/loss": losses['malyalam'],
                    "marathi/loss": losses['marathi'],
                    "odia/loss": losses['odia'],
                    "metrics/train_tokens_per_second": avg_train_tokens_per_second,
                    "metrics/test_tokens_per_second": avg_test_tokens_per_second,
                    "metrics/param_update_time_ms": avg_param_update_time * 1000,
                    "metrics/data_load_time_ms": np.mean(metrics['data_load_time']) * 1000,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                })
                
            if always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    
        if iter_num == 0 and eval_only:
            break

        train_tokens_this_iter = 0
        param_update_start = None
        
        # Forward-backward update with tqdm progress tracking
        for micro_step in tqdm(range(gradient_accumulation_steps), leave=False, desc="Micro Steps"):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                
            # Time the forward pass
            batch_start = time.time()
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
                
            # Track tokens processed in this forward pass
            tokens_in_batch = X.size(0) * X.size(1)  # batch_size * sequence_length
            train_tokens_this_iter += tokens_in_batch
            
            # Get next batch with timing
            data_load_start = time.time()
            X, Y = get_batch('train')
            data_load_end = time.time()
            metrics['data_load_time'].append(data_load_end - data_load_start)
            
            # Time the backward pass
            backward_start = time.time()
            scaler.scale(loss).backward()
            backward_end = time.time()
            
            batch_end = time.time()
            batch_duration = batch_end - batch_start
            metrics['batch_processing_time']['train'].append(batch_duration)
            
            # If this is the last micro step, we'll update parameters
            if micro_step == gradient_accumulation_steps - 1:
                param_update_start = time.time()
        
        # Calculate tokens per second for training (forward pass)
        batch_tokens_per_second = train_tokens_this_iter / (batch_duration * gradient_accumulation_steps)
        metrics['train_tokens_per_second'].append(batch_tokens_per_second)
        total_tokens_processed += train_tokens_this_iter
        
        # Clip gradients if required
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step with timing
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Measure parameter update time
        if param_update_start is not None:
            param_update_end = time.time()
            param_update_time = param_update_end - param_update_start
            metrics['param_update_time'].append(param_update_time)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            # Print detailed performance metrics every log_interval
            print_performance_metrics(iter_num, total_tokens_processed)
            
        local_iter_num += 1

        # Termination condition
        if iter_num >= max_iters:
            break

    if ddp:
        destroy_process_group()

    # Final performance summary
    if master_process:
        print("\n" + "="*50)
        print("FINAL PERFORMANCE SUMMARY:")
        print_performance_metrics(iter_num, total_tokens_processed)
except KeyboardInterrupt:
    pass
finally:
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

from data_loader import FastBinaryDataLoader

# 1st iter dataset
# train_bin_path = 'bin_data/train.bin'
# test_bin_paths = ['./bin_data/Hindi_test.bin',
#  './bin_data/gujrati_test.bin',
#  './bin_data/kannada_test.bin',
#  './bin_data/mar_test.bin',
#  './bin_data/mlym_test.bin',
#  './bin_data/orya_test.bin',]

# 2nd Iter dataset
paths = ['gujarati_processed_test.bin',   'kannada_processed_test.bin'     ,'marathi_processed_test.bin'   ,'odia_processed_train.bin',
'gujarati_processed_train.bin',  'kannada_processed_train.bin',    'marathi_processed_train.bin',
'hindi_processed_test.bin',      'malayalam_processed_test.bin'  , 'hindi_processed_train.bin' ,    'malayalam_processed_train.bin' , 'odia_processed_test.bin']
import numpy as np
total_tokens = 0
for path in paths:
        print(path,end=" ")
        path = './bin_data/' + path
        
        num_test_batches = len()
        total_tokens += num_test_batches * 8 * 1024
        print(": ",num_test_batches)
        
print(total_tokens)
import sys;
sys.exit()
# exit()
from tqdm import tqdm
import torch

def check_tensor(tensor, name):
    """Comprehensive tensor checking for NaNs and other issues."""
    if tensor is None:
        print(f"WARNING: {name} is None")
        return False
    
    if torch.isnan(tensor).any():
        print(f"WARNING: {name} contains NaN")
        return False
    
    if torch.isinf(tensor).any():
        print(f"WARNING: {name} contains Inf")
        return False
    
    return True

train_loader = FastBinaryDataLoader(
        data_path=train_bin_path, 
        batch_size=4, 
        shuffle=False,  # No need to shuffle test data
        sequence_length=256, 
        buffer_size=250_000, 
        prefetch=True  # No need to prefetch in test
        )

epoch_iterator = tqdm(
        enumerate(train_loader), 
        total=len(train_loader), 
        desc=f"Epoch 1", 
        position=1, 
        leave=False
)
            
total_loss = 0
for batch_idx, batch in epoch_iterator:

        inputs = torch.from_numpy(batch).to(dtype=torch.long)
        
        # Validate input tensor
        if not check_tensor(inputs, "Input Tensor"):
                print(f"Skipping batch {batch_idx} due to input tensor issues")
                continue
        
        # Create labels
        labels = inputs.clone()
        labels[:, :-1] = inputs[:, 1:]
        labels[:, -1] = inputs[:, -1]
        
        # Forward pass with additional checks
        # outputs = self.model_engine(inputs, labels=labels)
        # loss = outputs.loss / self.config['gradient_accumulation_steps']

        # Check loss before backward pass
        # if not check_tensor(loss, "Loss Tensor"):
        # print(f"Loss is NaN at batch {batch_idx}")
        # Optional: print more diagnostic information
        print("Batch inputs statistics:")
        print(f"Min: {inputs.min()}, Max: {inputs.max()}")
        print(f"Mean: {inputs.float().mean()}, Std: {inputs.float().std()}")
        # continue

        # Backward pass with gradient clipping
        # if hasattr(self.model_engine, 'backward'):
        # self.model_engine.backward(loss)
        # else:
        # loss.backward()
        
        # # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # total_loss += loss.item()
        break

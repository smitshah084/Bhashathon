"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

import sentencepiece as spm
model_path = "Preprocessing/the_10M/the_10M.model"
sp = spm.SentencePieceProcessor()
sp.load(model_path)
# def decode_tokens(sp: spm.SentencePieceProcessor, tokens: list) -> str:
#     try:
#         if tokens and isinstance(tokens[0], int):
#             return sp.decode_ids(tokens)
#         else:
#             return sp.decode_pieces(tokens)
#     except Exception as e:
#         print(f"Error decoding tokens: {str(e)}")
#         return ""

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
# load_meta = False
# if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
#     meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
#     load_meta = os.path.exists(meta_path)
# if load_meta:
#     print(f"Loading meta from {meta_path}...")
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     # TODO want to make this more general to arbitrary encoder/decoder schemes
#     stoi, itos = meta['stoi'], meta['itos']
#     encode = lambda s: [stoi[c] for c in s]
#     decode = lambda l: [itos[i] for i in l]
# else:
#     # ok let's assume gpt-2 encodings by default
#     print("No meta.pkl found, assuming GPT-2 encodings...")
#     enc = tiktoken.get_encoding("gpt2")
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#     decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
starts = [
    # Hindi
    "आज का दिन बहुत सुन्दर है ",
    "मुझे लगता है कि यह किताब बहुत अच्छी होगी ",
    "यह कहानी बहुत रोचक है ",
    
    # Gujarati
    "આજે આકાશ એકદમ નિખરેલું છે ",
    "મારે તમને એક રસપ્રદ વાત કહેવી છે ",
    "આજનું હવામાન ખુબ શાંત છે ",
    
    # Odia
    "ଆଜିର ଦିନଟି ବହୁତ ହସ୍ତସ୍ନିଗ୍ଧ ଅଟେ ",
    "ମୁଁ ଏକ ନୂତନ ଗଳ୍ପ ଆରମ୍ଭ କରିବାକୁ ଯାଉଛି ",
    "ଆଜି ଆମେ ଏକ ନୂଆ ଅନୁଭବ ଅଜିର୍ଣ୍ଣ କରିବୁ ",
    
    # Marathi
    "आजचा दिवस खूप सुंदर आहे ",
    "मला असं वाटतं की हा चित्रपट अप्रतिम आहे ",
    "ही गोष्ट ऐकून मला खूप आनंद झाला ",
    
    # Malayalam
    "ഇന്ന് ആകാശം വെട്ടിച്ചമഞ്ഞിരിക്കുന്നു ",
    "ഞാൻ ഒരു പുതിയ കഥ എഴുതാൻ പോകുന്നു ",
    "ഇന്നത്തെ കാലാവസ്ഥ വളരെ മനോഹരമാണ് ",
    
    # Kannada
    "ಇಂದು ನಭಸ್ಸು ಬಹಳ ಹೊಳಪಾಗಿದೆ ",
    "ನಾನು ಇಂದು ಹೊಸ ಕಥೆ ಆರಂಭಿಸುತ್ತಿದ್ದೇನೆ ",
    "ಇಲ್ಲಿನ ವಾತಾವರಣ ಬಹಳ ಸಂತೋಷಕರವಾಗಿದೆ "
]

for start in starts:
    start_ids = sp.encode_as_ids(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    print(start,end=" # ")
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                txt = sp.decode_ids(y[0].tolist())
                txt = str(txt)
                print(txt)
                print('---------------')

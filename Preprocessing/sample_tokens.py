import os
import logging
import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List, Dict, Optional
import random
from tqdm import tqdm
from time import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("sample_tokens.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_on_sample(sample_file: str = "../bin_data/train.bin", num_samples: int = 10) -> None:
        """
        Test tokenizer on a small sample from each language file.
        This is more memory efficient than loading all test sentences.
        """
        model_path = f"./indic_tokenizer_100M/indic_tokenizer_100M.model"
        
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return
        
        try:
            # Load the model
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            
            logger.info(f"Loaded model with vocabulary size: {sp.get_piece_size()}")
            
            # Sample a few sentences from each language file
            for file_path in ['../txt_data/Hindi.txt', '../txt_data/gujrati.txt', '../txt_data/kannada.txt', '../txt_data/mar.txt', '../txt_data/mlym.txt', '../txt_data/orya.txt']:
                print(file_path)
                ratio = 0
                try:
                    # Read a small sample
                    samples = []
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = []
                        for _ in range(num_samples * 10):  # Read a limited number of lines
                            line = f.readline()
                            if not line:
                                break
                            if line.strip():
                                lines.append(line.strip())
                        
                        # Sample randomly if we have enough lines
                        if len(lines) > num_samples:
                            indices = random.sample(range(len(lines)), num_samples)
                            samples = [lines[i] for i in indices]
                        else:
                            samples = lines
                    
                    # Test each sample
                    for i, sentence in enumerate(samples[:num_samples]):
                        if len(sentence) > 100:  # Truncate long sentences for display
                            sentence = sentence[:100] + "..."
                            
                        tokens = sp.encode_as_pieces(sentence)
                        ids = sp.encode_as_ids(sentence)
                        
                        # logger.info(f"Original: {len(sentence)} # Tokens: {len(tokens)} ratio {len(tokens)/len(sentence):.2f}")
                        ratio += len(tokens)/len(sentence)
                        # logger.info(f"# Tokens: {len(tokens)}")
                
                        
                except Exception as e:
                    logger.error(f"Error testing on {str(e)}")
                    
                ratio /= 10
                print(f"{file_path}: {ratio:.2f}")
                
        except Exception as e:
            logger.error(f"Error testing tokenizer: {str(e)}")
            

test_on_sample()
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultilingualTokenizer:
    """
    A class to build and train a multilingual tokenizer using SentencePiece.
    Optimized for low-memory systems.
    """
    
    def __init__(
        self,
        input_files: List[str],
        output_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        input_sentence_size: int = 1000000,  # Reduced from 5M to 1M
        shuffle_input_sentence: bool = True,
        normalization_rule_name: str = "nmt_nfkc_cf",
        num_threads: int = 4,
        split_by_whitespace: bool = True,
        split_by_number: bool = True,
        split_digits: bool = True,
        treat_whitespace_as_suffix: bool = False,
        languages: Optional[Dict[str, int]] = None,
        chunk_size: int = 100000  # New parameter for chunking
    ):
        """Initialize with lower default values for memory-sensitive parameters"""
        self.input_files = input_files
        self.output_prefix = output_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.input_sentence_size = input_sentence_size
        self.shuffle_input_sentence = shuffle_input_sentence
        self.normalization_rule_name = normalization_rule_name
        self.num_threads = num_threads
        self.split_by_whitespace = split_by_whitespace
        self.split_by_number = split_by_number
        self.split_digits = split_digits
        self.treat_whitespace_as_suffix = treat_whitespace_as_suffix
        self.languages = languages or {}
        self.chunk_size = chunk_size
        
        # Map file names to language keys
        self.file_to_lang = {
            'hindi_processed.txt': 'Hindi', 
            'gujarati_processed.txt': 'Gujarati',
            'kannada_processed.txt': 'Kannada',
            'marathi_processed.txt': 'Marathi',
            'malayalam_processed.txt': 'Malayalam',
            'odia_processed.txt': 'Odia'
        }          
        # self.file_line_counts = {
        #     '../../txt_data/kannada.txt':62_137_119,
        #     '../../txt_data/Hindi.txt':69_702_576,
        #     '../../txt_data/gujrati.txt'  : 48_219_331,
        #     '../../txt_data/odia.txt'  : 2_382_568,
        #     '../../txt_data/marathi.txt': 40_369_165,
        #     '../../txt_data/malayalam.txt'  : 64_879_852,
        # }
        # on previous iter
        # 1_138_360 lines from Marathi
        # 2_635_431 lines from Kannada
        # 1_584_919 lines from Hindi
        # 2_079_062 lines from Malayalam
        # 457_540 lines from Odia
        # 2_104_685 lines from Gujarati
        
        # Create output directory if it doesn't exist
        Path(os.path.dirname(output_prefix)).mkdir(parents=True, exist_ok=True)
        
    def check_input_files(self) -> bool:
        """Check if all input files exist."""
        for file_path in self.input_files:
            if not os.path.exists(file_path):
                logger.error(f"Input file does not exist: {file_path}")
                return False
        return True
    
    def count_lines(self, file_path: str) -> int:
        """Count lines in a file efficiently."""
        print("COUNTING LINES OF ",file_path)
        try:
            # Use wc command for efficient line counting
            import subprocess
            result = subprocess.run(['wc', '-l', file_path], 
                                   capture_output=True, text=True)
            o = int(result.stdout.split()[0])
            print("LINE COUNT of",file_path," :",o)
            return o
        except:
            # Fallback method if wc isn't available
            count = 0
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in f:
                    count += 1
            return count
    
    def prepare_training_file(self, combined_file: str = None) -> str:
        """
        Prepare a training file with memory-efficient processing.
        Uses reservoir sampling and chunking for large files.
        """
        print("PREPARING COMBINED TRAINING FILE START")
        if combined_file is None:
            combined_file = f"{self.output_prefix}_combined.txt"
            
        if os.path.exists(combined_file):
            logger.info(f"Combined file {combined_file} already exists. Skipping preparation.")
            return combined_file
        
        logger.info(f"Preparing training data from {len(self.input_files)} files")
        
        # Calculate target samples per language based on distribution
        total_size = sum(self.languages.values())
        target_samples = self.input_sentence_size
        samples_per_lang = {}
        
        for lang, size in self.languages.items():
            samples_per_lang[lang] = int((size / total_size) * target_samples)
            logger.info(f"Target samples for {lang}: {samples_per_lang[lang]}")
        
        # Process each file separately with reservoir sampling
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for file_path in self.input_files:
                print("PREPARING ",file_path," FOR TRAINING")
                base_name = os.path.basename(file_path)
                lang_name = self.file_to_lang.get(base_name, Path(file_path).stem)
                
                if lang_name not in samples_per_lang:
                    logger.warning(f"No sampling info for {base_name}, skipping")
                    continue
                
                num_samples = samples_per_lang.get(lang_name, 1000)
                
                # Get line count for progress reporting
                try:
                    # total_lines = self.count_lines(file_path)
                    total_lines = self.languages[lang_name]
                    logger.info(f"Processing {file_path} with {total_lines} lines")
                except Exception as e:
                    logger.warning(f"Couldn't count lines in {file_path}: {e}")
                    total_lines = None
                
                # Perform reservoir sampling in chunks to save memory
                reservoir = []
                lines_processed = 0
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    # Process file in chunks
                    pbar = tqdm(total=total_lines, desc=f"Sampling {lang_name}")
                    
                    for line in infile:
                        line = line.strip()
                        if not line:
                            continue
                            
                        lines_processed += 1
                        pbar.update(1)
                        
                        # Simple reservoir sampling
                        if len(reservoir) < num_samples:
                            reservoir.append(line)
                        else:
                            # Randomly replace elements with decreasing probability
                            j = random.randint(0, lines_processed - 1)
                            if j < num_samples:
                                reservoir[j] = line
                        
                        # Periodically clear memory
                        if lines_processed % 100000 == 0:
                            logger.debug(f"Processed {lines_processed} lines from {lang_name}")
                    
                    pbar.close()
                
                # Shuffle and write the reservoir to the output file
                random.shuffle(reservoir)
                for line in reservoir:
                    outfile.write(f"{line}\n")
                
                logger.info(f"Sampled {len(reservoir)} lines from {lang_name}")
        
        logger.info(f"Combined training data created at {combined_file}")
        return combined_file
    
    def train(self, combined_file: str = None) -> bool:
        """Train the SentencePiece tokenizer with memory-efficient settings."""
        if not self.check_input_files():
            return False
        
        # Prepare training data
        if combined_file is None:
            combined_file = self.prepare_training_file()
        
        # Build command line arguments for sentencepiece trainer
        train_args = {
            "input": combined_file,
            "model_prefix": self.output_prefix,
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "input_sentence_size": self.input_sentence_size,
            "shuffle_input_sentence": self.shuffle_input_sentence,
            "normalization_rule_name": self.normalization_rule_name,
            "num_threads": self.num_threads,
            "split_by_whitespace": self.split_by_whitespace,
            "split_by_number": self.split_by_number,
            "split_digits": self.split_digits,
            "treat_whitespace_as_suffix": self.treat_whitespace_as_suffix,
            "user_defined_symbols": "<pad>,<s>,</s>,<mask>",
            "unk_id": 3,
            "bos_id": 1,
            "eos_id": 2,
            "pad_id": 0,
            "max_sentence_length": 4192,  # Limit very long sentences
        }
        
        # Convert boolean flags to string
        for k, v in train_args.items():
            if isinstance(v, bool):
                train_args[k] = "true" if v else "false"
        
        # Log training parameters
        logger.info(f"Training tokenizer with parameters: {train_args}")
        
        try:
            # Train the model
            spm.SentencePieceTrainer.train(**train_args)
            
            # Verify model files were created
            model_path = f"{self.output_prefix}.model"
            vocab_path = f"{self.output_prefix}.vocab"
            
            if os.path.exists(model_path) and os.path.exists(vocab_path):
                logger.info(f"Training completed successfully. Model saved to {model_path}")
                # Clean up combined file to save space
                if os.path.exists(combined_file) and combined_file != combined_file:
                    os.remove(combined_file)
                    logger.info(f"Removed temporary combined file: {combined_file}")
                return True
            else:
                logger.error("Training failed: model files not created")
                return False
                
        except Exception as e:
            logger.error(f"Error during tokenizer training: {str(e)}")
            return False
    
    def test_on_sample(self, sample_file: str = None, num_samples: int = 10) -> None:
        """
        Test tokenizer on a small sample from each language file.
        This is more memory efficient than loading all test sentences.
        """
        model_path = f"{self.output_prefix}.model"
        
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return
        
        try:
            # Load the model
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            
            logger.info(f"Loaded model with vocabulary size: {sp.get_piece_size()}")
            
            # Sample a few sentences from each language file
            for file_path in self.input_files:
                base_name = os.path.basename(file_path)
                lang_name = self.file_to_lang.get(base_name, Path(file_path).stem)
                logger.info(f"Testing on {lang_name} samples:")
                
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
                        
                        logger.info(f"\n{lang_name} Sample {i+1}:")
                        logger.info(f"Original: {sentence}")
                        logger.info(f"Tokens: {tokens[:20]}{'...' if len(tokens)>20 else ''}")
                        logger.info(f"# Tokens: {len(tokens)}")
                        logger.info(f"# original: {len(sentence)}")
                        
                except Exception as e:
                    logger.error(f"Error testing on {lang_name}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error testing tokenizer: {str(e)}")


def main():
    """Main function with memory-optimized defaults."""
    parser = argparse.ArgumentParser(description="Train a multilingual SentencePiece tokenizer (memory-optimized)")
    
    parser.add_argument(
        "--output_prefix", 
        type=str, 
        default="multilingual_tokenizer",
        help="Prefix for the output model and vocabulary files"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=50304,
        help="Size of the vocabulary to be learned"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="Type of model (unigram, bpe, char, or word)"
    )
    parser.add_argument(
        "--character_coverage", 
        type=float, 
        default=0.9995,
        help="Character coverage percentage"
    )
    parser.add_argument(
        "--input_sentence_size", 
        type=int, 
        default=1000000,  # Reduced default
        help="Number of sentences to use for training"
    )
    parser.add_argument(
        "--num_threads", 
        type=int, 
        default=16,
        help="Number of threads to use for training"
    )
    
    args = parser.parse_args()
    input_files = ["../"+str(p) for p in Path("preprocessed/").glob("*.txt")]
    
    # Language distribution data
    language_data = {
        "Gujarati": 21770185,
        "Hindi": 18800523,
        "Kannada": 39910121,
        "Malayalam": 43789448,
        "Marathi": 23720708,
        "Odia": 622911
    }
# LINE COUNT of ../../txt_data/gujrati.txt  : 48_219_331
# LINE COUNT of ../../txt_data/Hindi.txt  : 69_702_576
# LINE COUNT of ../../txt_data/marathi.txt  : 40369165
# LINE COUNT of ../../txt_data/kannada.txt  : 62_137_119
# LINE COUNT of ../../txt_data/odia.txt  : 2382568
# LINE COUNT of ../../txt_data/malayalam.txt  : 64_879_852
# ./odia.txt: 20_104_412 words, 125_319_425 chars                                                                                                                                        
# ./marathi.txt: 529_202_722 words, 3_574_712_337 chars                                                                                                                                   
# ./gujrati.txt: 736_913_387 words, 4_328_510_189 chars                                                                                                                                   
# ./kannada.txt: 692_278_525 words, 5_660_951_964 chars                                                                                                                                   
# ./malayalam.txt: 675_825_752 words, 6_673_719_858 chars                                                                                                                                 
# ./Hindi.txt: 1_776_948_988 words, 9_096_419_775 chars 
    
    dir_path = args.output_prefix
    # while not os.path.isdir(dir_path):
    #     try:
    #         dir_path[-1] = str(int(dir_path[-1]) + 1)
    #     except:
    #         dir_path += "_1"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    os.chdir(dir_path)
        
    tokenizer = MultilingualTokenizer(
        input_files=input_files,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        num_threads=args.num_threads,
        languages=language_data
    )
    # com_file = tokenizer.prepare_training_file()
    # for i in input_files:
    #     a = tokenizer.count_lines(i)
    #     print(i,a)
    # import sys; sys.exit()
    s = time()
    success = tokenizer.train()
    s = time() - s
    logger.info(f"Took {s} seconds for proccessing {args.input_sentence_size} sentences.")
    
    if success:
        tokenizer.test_on_sample()
    

if __name__ == "__main__":
    main()

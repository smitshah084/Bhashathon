#!/usr/bin/env python3
"""
Preprocessing pipeline for multilingual text data, specifically optimized for Indian languages.
This script handles various text normalization, cleaning, and filtering tasks to prepare
data for tokenizer training.
"""

import os
import re
import logging
import argparse
import unicodedata
import hashlib
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultilingualPreprocessor:
    """
    A class to preprocess multilingual text data for tokenizer training.
    Focuses on Indian languages with script-specific processing.
    """
    
    def __init__(
        self,
        input_files=None,
        output_dir="preprocessed",
        languages=None,
        num_threads=4,
        sample_size=None,
        deduplicate=True,
        clean_unicode=True,
        remove_low_quality=True,
        script_filtering=True
    ):
        """Initialize the preprocessor with configuration options."""
        self.input_files = input_files or []
        self.output_dir = output_dir
        self.num_threads = num_threads
        self.sample_size = sample_size
        self.deduplicate = deduplicate
        self.clean_unicode = clean_unicode
        self.remove_low_quality = remove_low_quality
        self.script_filtering = script_filtering
        
        # Default language mappings
        self.file_to_lang = languages or {
            'Hindi.txt': 'Hindi', 
            'gujrati.txt': 'Gujarati',
            'kannada.txt': 'Kannada',
            'marathi.txt': 'Marathi',
            'malayalam.txt': 'Malayalam',
            'odia.txt': 'Odia'
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Script ranges for different Indian languages
        self.script_ranges = {
            "Hindi": (0x0900, 0x097F),      # Devanagari
            "Gujarati": (0x0A80, 0x0AFF),   # Gujarati
            "Malayalam": (0x0D00, 0x0D7F),  # Malayalam
            "Kannada": (0x0C80, 0x0CFF),    # Kannada
            "Marathi": (0x0900, 0x097F),    # Devanagari (like Hindi)
            "Odia": (0x0B00, 0x0B7F)        # Oriya
        }
        
        # Language-specific normalization mappings
        self.normalization_maps = self._init_normalization_maps()
    
    def _init_normalization_maps(self):
        """Initialize language-specific character normalization mappings."""
        maps = {}
        
        # Hindi/Devanagari normalization
        hindi_map = {
            '॰': '.',           # Devanagari abbreviation to period
            '\u200c': '',       # Zero width non-joiner
            '\u200d': '',       # Zero width joiner
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'  # Optional: normalize digits
        }
        maps["Hindi"] = hindi_map
        maps["Marathi"] = hindi_map  # Marathi uses Devanagari too
        
        # Gujarati normalization
        gujarati_map = {
            '\u0acd\u0a3e': '\u0abe',  # Handle specific Gujarati conjuncts
            '\u200c': '',              # Zero width non-joiner
            '\u200d': ''               # Zero width joiner
        }
        maps["Gujarati"] = gujarati_map
        
        # Malayalam normalization
        malayalam_map = {
            '\u200c': '',       # Zero width non-joiner
            '\u200d': '',       # Zero width joiner
            # Add specific Malayalam normalizations as needed
        }
        maps["Malayalam"] = malayalam_map
        
        # Kannada normalization
        kannada_map = {
            '\u200c': '',       # Zero width non-joiner
            '\u200d': '',       # Zero width joiner
            # Add specific Kannada normalizations as needed
        }
        maps["Kannada"] = kannada_map
        
        # Odia normalization
        odia_map = {
            '\u200c': '',       # Zero width non-joiner
            '\u200d': '',       # Zero width joiner
            # Add specific Odia normalizations as needed
        }
        maps["Odia"] = odia_map
        
        return maps
    
    def process_all_files(self):
        """Process all input files using parallel execution."""
        processed_files = []
        
        if not self.input_files:
            logger.error("No input files specified.")
            return processed_files
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_file = {}
            
            for file_path in self.input_files:
                if not os.path.exists(file_path):
                    logger.warning(f"Input file does not exist: {file_path}")
                    continue
                
                base_name = os.path.basename(file_path)
                language = self.file_to_lang.get(base_name, Path(file_path).stem)
                
                output_file = os.path.join(self.output_dir, f"{language.lower()}_processed.txt")
                future = executor.submit(
                    self.process_file, 
                    file_path, 
                    language, 
                    output_file
                )
                future_to_file[future] = (file_path, language, output_file)
            
            # Process results as they complete
            for future in tqdm(future_to_file, total=len(future_to_file), desc="Processing files"):
                file_path, language, output_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(output_file)
                        logger.info(f"Successfully processed {language} data: {output_file}")
                    else:
                        logger.error(f"Failed to process {language} data: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Generate statistics for processed files
        for file_path in processed_files:
            language = Path(file_path).stem.split('_')[0].capitalize()
            self.generate_file_statistics(file_path, language)
        
        return processed_files
    
    def process_file(self, file_path, language, output_file):
        """Process a single file with all preprocessing steps."""
        logger.info(f"Processing {language} file: {file_path}")
        
        try:
            # Output file for processed text
            temp_file = output_file + ".temp"
            
            # Create an empty processed file
            with open(temp_file, 'w', encoding='utf-8') as out_f:
                pass
                
            # Process the file in chunks to save memory
            total_lines = 0
            processed_lines = 0
            skipped_lines = 0
            
            chunk_size = 10000  # Process 10k lines at a time
            seen_hashes = set() if self.deduplicate else None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                with open(temp_file, 'w', encoding='utf-8') as out_f:
                    
                    # Read and process file in chunks
                    chunk = []
                    for line in tqdm(in_f, desc=f"Processing {language}", unit="lines"):
                        total_lines += 1
                        
                        chunk.append(line)
                        if len(chunk) >= chunk_size:
                            processed, skipped = self._process_chunk(chunk, out_f, language, seen_hashes)
                            processed_lines += processed
                            skipped_lines += skipped
                            chunk = []
                    
                    # Process any remaining lines
                    if chunk:
                        processed, skipped = self._process_chunk(chunk, out_f, language, seen_hashes)
                        processed_lines += processed
                        skipped_lines += skipped
            
            # If we're sampling, perform the sampling now
            if self.sample_size and self.sample_size > 0:
                sampled_file = output_file
                self._sample_lines(temp_file, sampled_file, self.sample_size)
                os.remove(temp_file)
            else:
                # Just rename the temp file to the final output
                os.rename(temp_file, output_file)
            
            logger.info(f"Completed processing {language}: {total_lines} lines read, "
                       f"{processed_lines} lines kept, {skipped_lines} lines skipped")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {language} file: {str(e)}")
            return False
    
    def _process_chunk(self, chunk, out_file, language, seen_hashes=None):
        """Process a chunk of lines from the input file."""
        processed_count = 0
        skipped_count = 0
        
        for line in chunk:
            processed_line = self.preprocess_line(line, language)
            
            # Skip empty lines
            if not processed_line:
                skipped_count += 1
                continue
                
            # Optionally deduplicate using hashing
            if seen_hashes is not None:
                line_hash = hashlib.md5(processed_line.encode('utf-8')).hexdigest()
                if line_hash in seen_hashes:
                    skipped_count += 1
                    continue
                seen_hashes.add(line_hash)
            
            # Write the processed line
            out_file.write(processed_line + '\n')
            processed_count += 1
                
        return processed_count, skipped_count
    
    def _sample_lines(self, input_file, output_file, sample_size):
        """Randomly sample lines from input file using reservoir sampling."""
        try:
            # Count lines
            with open(input_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            # If file has fewer lines than sample size, just copy it
            if line_count <= sample_size:
                os.rename(input_file, output_file)
                return
            
            # Perform reservoir sampling
            reservoir = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < sample_size:
                        reservoir.append(line)
                    else:
                        # Randomly replace elements with decreasing probability
                        j = random.randint(0, i)
                        if j < sample_size:
                            reservoir[j] = line
            
            # Write sampled lines to output
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in reservoir:
                    f.write(line)
                    
            logger.info(f"Sampled {sample_size} lines from {input_file} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error sampling lines: {str(e)}")
            # If sampling fails, just use the original file
            if os.path.exists(input_file):
                os.rename(input_file, output_file)
    
    def preprocess_line(self, line, language):
        """Apply all preprocessing steps to a single line of text."""
        # Skip empty lines
        if not line.strip():
            return ""
        
        # Basic cleaning
        line = self.remove_html(line)
        line = self.clean_whitespace(line)
        
        # Remove control characters except newlines and tabs
        line = ''.join(ch for ch in line if ch == '\n' or ch == '\t' or ord(ch) >= 32)
        
        # Unicode normalization if enabled
        if self.clean_unicode:
            line = self.normalize_unicode(line)
            line = self.normalize_script(line, language)
        
        # Quality filtering if enabled
        if self.remove_low_quality and not self.is_quality_content(line):
            return ""
            
        # Script validation if enabled
        if self.script_filtering and not self.is_target_script(line, language):
            return ""
        
        return line.strip()
    
    def normalize_unicode(self, text):
        """Normalize unicode text to NFKC form."""
        return unicodedata.normalize('NFKC', text)
    
    def normalize_script(self, text, language):
        """Apply language-specific normalization."""
        if language not in self.normalization_maps:
            return text
            
        # Apply character substitutions
        for src, tgt in self.normalization_maps[language].items():
            text = text.replace(src, tgt)
            
        return text
    
    def remove_html(self, text):
        """Remove HTML/XML tags from text."""
        return re.sub(r'<[^>]+>', '', text)
    
    def clean_whitespace(self, text):
        """Normalize whitespace in text."""
        # Replace tabs, newlines with spaces
        text = re.sub(r'[\t\n\r]+', ' ', text)
        # Collapse multiple spaces into single space
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def is_quality_content(self, text):
        """Filter out low-quality or nonsensical text."""
        # Remove if too short
        if len(text) < 5:
            return False
        
        # Remove if too repetitive (e.g., "aaaaaa")
        if re.search(r'(.)\1{5,}', text):
            return False
        
        # Remove if mostly punctuation or symbols
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return False
            
        return True
    
    def is_target_script(self, text, language):
        """Check if text is primarily in the expected script."""
        if language not in self.script_ranges:
            return True
            
        start, end = self.script_ranges[language]
        
        # Count characters in expected script range
        char_count = 0
        script_chars = 0
        
        for c in text:
            if c.isalpha():
                char_count += 1
                if start <= ord(c) <= end:
                    script_chars += 1
        
        # If no alphabetic characters, accept it (could be numbers, punctuation)
        if char_count == 0:
            return True
            
        # If at least 60% characters are in target script, accept the line
        return (script_chars / char_count) > 0.6
    
    def segment_sentences(self, text):
        """Split text into sentences for better sampling."""
        # For Indic languages, handle both Western and Indic punctuation
        sentences = re.split(r'[।.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def generate_file_statistics(self, file_path, language):
        """Generate statistics for a processed file."""
        if not os.path.exists(file_path):
            logger.warning(f"Cannot generate statistics for non-existent file: {file_path}")
            return
            
        try:
            line_count = 0
            word_count = 0
            char_count = 0
            char_freq = {}
            word_lengths = []
            sentence_lengths = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Character stats
                    for char in line:
                        char_count += 1
                        char_freq[char] = char_freq.get(char, 0) + 1
                    
                    # Word stats
                    words = line.strip().split()
                    word_count += len(words)
                    word_lengths.extend([len(w) for w in words])
                    
                    # Sentence stats
                    sentences = self.segment_sentences(line)
                    sentence_lengths.extend([len(s.split()) for s in sentences])
            
            # Generate summary statistics
            stats = {
                "file_name": os.path.basename(file_path),
                "language": language,
                "line_count": line_count,
                "word_count": word_count,
                "char_count": char_count,
                "vocab_size": len(set(" ".join(open(file_path, 'r', encoding='utf-8').readlines()).split())),
                "unique_chars": len(char_freq),
                "avg_word_length": sum(word_lengths) / max(len(word_lengths), 1),
                "avg_sentence_length": sum(sentence_lengths) / max(len(sentence_lengths), 1) if sentence_lengths else 0,
                "top_chars": sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            }
            
            # Log statistics
            logger.info(f"Statistics for {language}:")
            logger.info(f"  Lines: {stats['line_count']:,}")
            logger.info(f"  Words: {stats['word_count']:,}")
            logger.info(f"  Characters: {stats['char_count']:,}")
            logger.info(f"  Vocabulary size: {stats['vocab_size']:,}")
            logger.info(f"  Unique characters: {stats['unique_chars']}")
            logger.info(f"  Average word length: {stats['avg_word_length']:.2f}")
            logger.info(f"  Average sentence length: {stats['avg_sentence_length']:.2f}")
            
            # Write detailed statistics to a file
            stats_file = os.path.join(self.output_dir, f"{language.lower()}_stats.txt")
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"Statistics for {language} ({os.path.basename(file_path)}):\n")
                f.write(f"  Lines: {stats['line_count']:,}\n")
                f.write(f"  Words: {stats['word_count']:,}\n")
                f.write(f"  Characters: {stats['char_count']:,}\n")
                f.write(f"  Vocabulary size: {stats['vocab_size']:,}\n")
                f.write(f"  Unique characters: {stats['unique_chars']}\n")
                f.write(f"  Average word length: {stats['avg_word_length']:.2f}\n")
                f.write(f"  Average sentence length: {stats['avg_sentence_length']:.2f}\n\n")
                
                f.write("Top 20 most frequent characters:\n")
                for char, count in stats["top_chars"]:
                    if char in ['\n', '\t', '\r', ' ']:
                        char_display = f"'{ord(char)}'"
                    else:
                        char_display = f"'{char}'"
                    f.write(f"  {char_display}: {count:,}\n")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics for {file_path}: {str(e)}")
            return None

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Preprocess multilingual text data for tokenizer training")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../txt_data/",
        help="Directory containing input text files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preprocessed",
        help="Directory to store preprocessed files"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads for parallel processing"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="If set, randomly sample this many lines from each input file"
    )
    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="Disable deduplication of lines"
    )
    parser.add_argument(
        "--no_unicode_clean",
        action="store_true",
        help="Disable Unicode normalization"
    )
    parser.add_argument(
        "--no_quality_filter",
        action="store_true",
        help="Disable quality filtering"
    )
    parser.add_argument(
        "--no_script_filter",
        action="store_true",
        help="Disable script-based filtering"
    )
    
    args = parser.parse_args()
    
    # Find input files
    if os.path.isdir(args.input_dir):
        input_files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                      if f.endswith('.txt')])
    else:
        input_files = [args.input_dir] if args.input_dir.endswith('.txt') else []
    
    if not input_files:
        print(f"No .txt files found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} input files: {[os.path.basename(f) for f in input_files]}")
    
    # Create preprocessor instance
    preprocessor = MultilingualPreprocessor(
        input_files=input_files,
        output_dir=args.output_dir,
        num_threads=args.num_threads,
        sample_size=args.sample_size,
        deduplicate=not args.no_deduplicate,
        clean_unicode=not args.no_unicode_clean,
        remove_low_quality=not args.no_quality_filter,
        script_filtering=not args.no_script_filter
    )
    
    # Process all files
    processed_files = preprocessor.process_all_files()
    
    if processed_files:
        print(f"Successfully processed {len(processed_files)} files:")
        for file_path in processed_files:
            print(f"  - {file_path}")
        print(f"Statistics are available in the {args.output_dir} directory")
    else:
        print("No files were successfully processed")

if __name__ == "__main__":
    main()

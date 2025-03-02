import os
import numpy as np
import sentencepiece as spm
from typing import Dict, List, Tuple, Iterator
import random
from tqdm import tqdm
import mmap
import multiprocessing as mp
import psutil
from functools import partial
import json
import threading
import time
import gc

# Define worker functions outside of classes to make them picklable

def calculate_line_offsets_worker(args):
    """Worker function to calculate line offsets for a file"""
    lang, file_path = args
    offsets = [0]
    try:
        with open(file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            line = mm.readline()
            while line:
                offsets.append(mm.tell())
                line = mm.readline()
            mm.close()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return lang, []
    return lang, offsets[:-1]  # Remove the last offset which is EOF

def process_chunk_worker(args):
    """Process a chunk of text from a file based on line offsets"""
    file_path, offsets, tokenizer_model_path, lang, is_test, sequence_length = args
    
    # Load tokenizer in each worker process
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_model_path)
    
    sequences = []
    current_tokens = np.array([], dtype=np.uint16)
    
    try:
        with open(file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            for offset in offsets:
                mm.seek(offset)
                line = mm.readline().decode('utf-8', errors='ignore').strip()
                
                # Tokenize line
                line_tokens = np.array(tokenizer.encode(line), dtype=np.uint16)
                current_tokens = np.append(current_tokens, line_tokens)
                
                # Create full sequences of length sequence_length
                while len(current_tokens) >= sequence_length:
                    sequence = current_tokens[:sequence_length]
                    current_tokens = current_tokens[sequence_length:]
                    sequences.append(sequence)
            
            mm.close()
    except Exception as e:
        print(f"Error processing chunk in {file_path}: {e}")
        return [], lang, is_test
    
    return sequences, lang, is_test

class ResourceMonitor:
    """Monitor and report system resource usage"""
    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = None
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
    
    def _monitor(self):
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_used_gb = memory_info.used / (1024**3)
            memory_percent = memory_info.percent
            
            self.stats['cpu_usage'].append(cpu_percent)
            self.stats['memory_usage'].append(memory_used_gb)
            self.stats['timestamps'].append(time.time() - self.start_time)
            
            elapsed = time.time() - self.start_time
            print(f"\r[{elapsed:.1f}s] CPU: {cpu_percent:.1f}% | Memory: {memory_used_gb:.1f}GB ({memory_percent:.1f}%)", end="")
            
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Print final summary
        if self.stats['cpu_usage']:
            avg_cpu = np.mean(self.stats['cpu_usage'])
            max_cpu = np.max(self.stats['cpu_usage'])
            avg_mem = np.mean(self.stats['memory_usage'])
            max_mem = np.max(self.stats['memory_usage'])
            elapsed = time.time() - self.start_time
            
            print("\n\nResource Usage Summary:")
            print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            print(f"Average CPU usage: {avg_cpu:.1f}%")
            print(f"Peak CPU usage: {max_cpu:.1f}%")
            print(f"Average memory usage: {avg_mem:.2f} GB")
            print(f"Peak memory usage: {max_mem:.2f} GB")

class MultilingualDatasetOptimized:
    def __init__(
        self, 
        data_dir: str,
        model_path: str,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        sequence_length: int = 1024,  # Updated to 1024
        num_workers: int = 6  # Optimized for n2-standard-8, leave 2 cores for main thread and system
    ):
        """
        Initialize the optimized multilingual dataset loader
        
        Args:
            data_dir: Directory containing the language text files
            model_path: Path to the SentencePiece model
            test_ratio: Ratio of data to use for testing
            random_seed: Random seed for reproducibility
            sequence_length: Length of each sequence (default: 1024)
            num_workers: Number of worker processes (default: 6 for n2-standard-8)
        """
        self.data_dir = data_dir
        self.test_ratio = test_ratio
        self.sequence_length = sequence_length
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Set up multiprocessing - optimized for n2-standard-8
        self.num_workers = num_workers
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_path)
        self.tokenizer_model_path = model_path
        
        # Get file paths
        self.file_paths = self._get_file_paths()
        
        # Calculate line offsets for each file (for efficient random access)
        self.file_offsets = {}
        
        # Resource monitor
        self.monitor = ResourceMonitor(interval=10)
        
    def _get_file_paths(self) -> Dict[str, str]:
        """Get the paths of all language files"""
        files = ['Hindi.txt', 'gujrati.txt', 'kannada.txt', 'mar.txt', 'mlym.txt', 'orya.txt']
        return {f.split('.')[0]: os.path.join(self.data_dir, f) for f in files}
    
    def calculate_offsets(self):
        """Calculate line offsets for all files"""
        print("Calculating line offsets for all files...")
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        file_pairs = [(lang, path) for lang, path in self.file_paths.items()]
        
        # Use multiprocessing to calculate offsets in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(calculate_line_offsets_worker, file_pairs)
        
        # Store the results
        for lang, offsets in results:
            self.file_offsets[lang] = offsets
            print(f"{lang}: {len(self.file_offsets[lang])} lines")
        
        elapsed = time.time() - start_time
        print(f"Offset calculation completed in {elapsed:.2f} seconds")
    
    def _create_stratified_split_indices(self, lang: str) -> Tuple[List[int], List[int]]:
        """Create stratified train/test split indices for a language"""
        offsets = self.file_offsets[lang]
        total_lines = len(offsets)
        
        # Generate shuffled indices
        indices = list(range(total_lines))
        random.shuffle(indices)
        
        # Split into train and test
        test_size = int(total_lines * self.test_ratio)
        test_indices = set(indices[:test_size])
        
        train_indices = [i for i in range(total_lines) if i not in test_indices]
        test_indices = list(test_indices)
        
        return train_indices, test_indices
    
    def _save_sequences_to_file(self, sequences: List[np.ndarray], file_path: str):
        """Save sequences to a binary file"""
        with open(file_path, 'ab') as f:
            for seq in sequences:
                seq.tofile(f)
    
    def save_processed_data(self, output_dir: str, chunk_size: int = 2000):  # Increased chunk size for better throughput
        """
        Process all data and save to:
        - One combined train file for all languages
        - Separate test files for each language
        
        Args:
            output_dir: Directory to save processed files
            chunk_size: Number of lines to process in each worker chunk (optimized)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create resource monitor
        self.monitor.start()
        
        # Calculate offsets if not done already
        if not self.file_offsets:
            self.calculate_offsets()
        
        # Estimate processing time
        total_lines = sum(len(self.file_offsets[lang]) for lang in self.file_paths.keys() if lang in self.file_offsets)
        estimated_lines_per_second = 5000  # Rough estimate, will be adjusted based on initial processing
        estimated_seconds = total_lines / estimated_lines_per_second
        print(f"Estimated processing time: {estimated_seconds:.2f} seconds ({estimated_seconds/60:.2f} minutes)")
        
        # Create train file
        train_file = os.path.join(output_dir, 'train.bin')
        open(train_file, 'wb').close()
        
        # Create test files for each language
        test_files = {}
        for lang in self.file_paths.keys():
            test_file = os.path.join(output_dir, f'{lang}_test.bin')
            test_files[lang] = test_file
            open(test_file, 'wb').close()
        
        # Save metadata
        metadata = {
            "sequence_length": self.sequence_length,
            "vocab_size": self.tokenizer.get_piece_size(),
            "dtype": "uint16",
            "languages": {},
            "train_test_ratio": 1.0 - self.test_ratio,
            "machine_specs": {
                "vcpus": 8,
                "memory_gb": 32,
                "workers": self.num_workers
            },
            "files": {
                "train": os.path.basename(train_file),
                "test": {lang: os.path.basename(file) for lang, file in test_files.items()}
            }
        }
        
        # Process each language
        train_sequence_count = 0
        test_sequence_counts = {lang: 0 for lang in self.file_paths.keys()}
        
        # Track processing speeds
        processing_start_time = time.time()
        lines_processed = 0
        
        for lang, file_path in tqdm(self.file_paths.items(), desc="Processing languages"):
            if lang not in self.file_offsets:
                print(f"Skipping {lang} - no offset data available")
                continue
                
            print(f"\nProcessing {lang}...")
            
            # Create stratified split
            train_indices, test_indices = self._create_stratified_split_indices(lang)
            
            # Save language stats in metadata
            metadata["languages"][lang] = {
                "total_lines": len(self.file_offsets[lang]),
                "train_lines": len(train_indices),
                "test_lines": len(test_indices)
            }
            
            # Create processing chunks (train)
            train_chunks = []
            for i in range(0, len(train_indices), chunk_size):
                chunk_indices = train_indices[i:i+chunk_size]
                chunk_offsets = [self.file_offsets[lang][idx] for idx in chunk_indices]
                train_chunks.append((file_path, chunk_offsets, self.tokenizer_model_path, lang, False, self.sequence_length))
            
            # Create processing chunks (test)
            test_chunks = []
            for i in range(0, len(test_indices), chunk_size):
                chunk_indices = test_indices[i:i+chunk_size]
                chunk_offsets = [self.file_offsets[lang][idx] for idx in chunk_indices]
                test_chunks.append((file_path, chunk_offsets, self.tokenizer_model_path, lang, True, self.sequence_length))
            
            # Process all chunks with multiprocessing
            all_chunks = train_chunks + test_chunks
            total_chunks = len(all_chunks)
            lines_in_chunks = len(train_indices) + len(test_indices)
            lines_processed += lines_in_chunks
            
            # Process in smaller batches to avoid memory issues
            # For n2-standard-8 with 32GB, we can process more chunks in parallel
            batch_size = min(total_chunks, self.num_workers * 3)
            
            for i in range(0, total_chunks, batch_size):
                current_chunks = all_chunks[i:i+batch_size]
                chunk_start_time = time.time()
                
                with mp.Pool(processes=self.num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_chunk_worker, current_chunks),
                        total=len(current_chunks),
                        desc=f"Processing chunks {i}-{min(i+batch_size, total_chunks)}/{total_chunks}"
                    ))
                
                # Save results to files
                for sequences, lang, is_test in results:
                    if not sequences:  # Skip empty sequences
                        continue
                        
                    if is_test:
                        self._save_sequences_to_file(sequences, test_files[lang])
                        test_sequence_counts[lang] += len(sequences)
                    else:
                        self._save_sequences_to_file(sequences, train_file)
                        train_sequence_count += len(sequences)
                
                # Calculate and report processing speed
                chunk_time = time.time() - chunk_start_time
                lines_in_batch = sum(len(chunks[1]) for chunks in current_chunks)
                lines_per_second = lines_in_batch / chunk_time
                
                print(f"Batch processing speed: {lines_per_second:.2f} lines/sec")
                
                # Update time estimate
                elapsed = time.time() - processing_start_time
                remaining_lines = total_lines - lines_processed
                estimated_remaining = remaining_lines / lines_per_second
                print(f"Estimated remaining time: {estimated_remaining:.2f} seconds ({estimated_remaining/60:.2f} minutes)")
                
                # Force garbage collection to free memory
                results = None
                gc.collect()
        
        # Update metadata with sequence counts and timing
        total_processing_time = time.time() - processing_start_time
        metadata["sequence_counts"] = {
            "train": train_sequence_count,
            "test": test_sequence_counts
        }
        metadata["processing_stats"] = {
            "total_time_seconds": total_processing_time,
            "lines_processed": lines_processed,
            "lines_per_second": lines_processed / total_processing_time
        }
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Stop resource monitor
        self.monitor.stop()
        
        print(f"\nSaved processed data to {output_dir}")
        print(f"Train file: {train_file} ({train_sequence_count} sequences)")
        for lang, count in test_sequence_counts.items():
            print(f"{lang} test file: {test_files[lang]} ({count} sequences)")
        
        print(f"\nTotal processing time: {total_processing_time:.2f} seconds ({total_processing_time/60:.2f} minutes)")
        print(f"Average processing speed: {lines_processed/total_processing_time:.2f} lines/second")


def shuffle_binary_file(file_path, sequence_length=1024, buffer_size=100000, output_path=None):
    """
    Shuffle the sequences in a binary file.
    Uses an external merge sort approach for memory efficiency.
    
    Args:
        file_path: Path to the binary file to shuffle
        sequence_length: Length of each sequence (updated to 1024)
        buffer_size: Number of sequences to load at once (increased for 32GB RAM)
        output_path: Path to save shuffled data (if None, overwrites original)
    """
    if output_path is None:
        output_path = file_path + ".shuffled"
    
    # Start resource monitor
    monitor = ResourceMonitor(interval=5)
    monitor.start()
    
    # Get file size
    file_size = os.path.getsize(file_path)
    n_sequences = file_size // (sequence_length * 2)  # uint16 = 2 bytes
    bytes_per_sequence = sequence_length * 2
    
    print(f"Shuffling {n_sequences} sequences...")
    
    # Calculate memory requirements
    memory_required = buffer_size * sequence_length * 2 / (1024**3)  # in GB
    print(f"Memory required for buffer: {memory_required:.2f} GB")
    
    start_time = time.time()
    
    # Create shuffled order of all sequences
    print("Creating shuffled sequence order...")
    order = np.arange(n_sequences)
    np.random.shuffle(order)
    
    # Process in chunks to avoid memory issues
    print("Writing shuffled data...")
    with open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        for i in tqdm(range(0, n_sequences, buffer_size), desc="Shuffling chunks"):
            # Determine chunk size
            chunk_size = min(buffer_size, n_sequences - i)
            
            # Allocate buffer for this chunk
            buffer = np.zeros((chunk_size, sequence_length), dtype=np.uint16)
            
            # Read sequences in shuffled order for this chunk
            chunk_start_time = time.time()
            for j in range(chunk_size):
                seq_idx = order[i + j]
                f_in.seek(seq_idx * bytes_per_sequence)
                data = np.fromfile(f_in, dtype=np.uint16, count=sequence_length)
                buffer[j] = data
            
            # Write shuffled chunk
            buffer.tofile(f_out)
            
            # Calculate and report processing speed
            chunk_time = time.time() - chunk_start_time
            seqs_per_second = chunk_size / chunk_time
            
            # Estimate remaining time
            elapsed = time.time() - start_time
            processed = i + chunk_size
            remaining = n_sequences - processed
            estimated_remaining = remaining / seqs_per_second
            
            print(f"\rProcessing speed: {seqs_per_second:.2f} seqs/sec | " +
                  f"Estimated remaining time: {estimated_remaining/60:.2f} min", end="")
    
    # If we're replacing the original file, rename the shuffled file
    if output_path != file_path:
        os.rename(output_path, file_path)
    
    total_time = time.time() - start_time
    monitor.stop()
    
    print(f"\nShuffled data saved to {file_path}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average processing speed: {n_sequences/total_time:.2f} sequences/second")

# Example usage:
if __name__ == "__main__":
    # Example configuration
    data_dir = "../txt_data"
    model_path = "./indic_tokenizer_100M/indic_tokenizer_100M.model"
    output_dir = "../bin_data"
    
    # Print system information
    print("System Information:")
    print(f"CPU: {psutil.cpu_count(logical=True)} logical cores, {psutil.cpu_count(logical=False)} physical cores")
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
    
    # Initialize and process the dataset
    dataset = MultilingualDatasetOptimized(
        data_dir=data_dir,
        model_path=model_path,
        test_ratio=0.05,
        sequence_length=1024,  # Using 1024 as requested
        num_workers=6  # Optimized for n2-standard-8
    )
    
    # Process and save all data to binary files
    dataset.save_processed_data(output_dir, chunk_size=2000)  # Larger chunk size for efficiency
    
    # Shuffle the combined training file for better training
    print("\nShuffling the combined training file...")
    shuffle_binary_file(
        file_path=os.path.join(output_dir, 'train.bin'),
        sequence_length=1024,  # Updated to 1024
        buffer_size=100000  # Increased for 32GB RAM
    )
    
    print("\nData processing complete!")
    print("To load the training data:")
    print("train_loader = FastBinaryDataLoader(")
    print(f"    data_path='{os.path.join(output_dir, 'train.bin')}',")
    print("    batch_size=64,")  # Increased batch size for n2-standard-8
    print("    shuffle=True,")
    print("    sequence_length=1024,")  # Updated to 1024
    print("    buffer_size=250000  # Optimized for 32GB RAM")
    print(")")
    
    print("\nTo evaluate on a specific language:")
    for lang in dataset.file_paths.keys():
        print(f"\n# For {lang} test data:")
        print(f"{lang}_test_loader = FastBinaryDataLoader(")
        print(f"    data_path='{os.path.join(output_dir, f'{lang}_test.bin')}',")
        print("    batch_size=64,")  # Increased batch size
        print("    shuffle=False,  # Usually no need to shuffle test data")
        print("    sequence_length=1024,")  # Updated to 1024
        print("    buffer_size=50000")  # Increased for 32GB RAM
        print(")")
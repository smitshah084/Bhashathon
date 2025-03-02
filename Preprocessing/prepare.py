import os
import numpy as np
import sentencepiece as spm
from typing import Dict, List, Tuple, Iterator
import random
from tqdm import tqdm
import mmap
import multiprocessing as mp
from functools import partial
import json
import threading

class MultilingualDatasetOptimized:
    def __init__(
        self, 
        data_dir: str,
        model_path: str,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        sequence_length: int = 512,
        num_workers: int = None
    ):
        """
        Initialize the optimized multilingual dataset loader
        
        Args:
            data_dir: Directory containing the language text files
            model_path: Path to the SentencePiece model
            test_ratio: Ratio of data to use for testing
            random_seed: Random seed for reproducibility
            sequence_length: Length of each sequence
            num_workers: Number of worker processes (defaults to CPU count - 1)
        """
        self.data_dir = data_dir
        self.test_ratio = test_ratio
        self.sequence_length = sequence_length
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Set up multiprocessing
        if num_workers is None:
            # Use CPU count - 1 to leave one core free for system processes
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = num_workers
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_path)
        self.tokenizer_model_path = model_path
        
        # Get file paths
        self.file_paths = self._get_file_paths()
        
        # Calculate line offsets for each file (for efficient random access)
        self.file_offsets = {}
        
    def _get_file_paths(self) -> Dict[str, str]:
        """Get the paths of all language files"""
        files = ['Hindi.txt', 'gujrati.txt', 'kannada.txt', 'mar.txt', 'mlym.txt', 'orya.txt']
        return {f.split('.')[0]: os.path.join(self.data_dir, f) for f in files}
    
    def _calculate_line_offsets(self, file_path: str) -> List[int]:
        """Calculate byte offsets for each line in a file"""
        offsets = [0]
        with open(file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            line = mm.readline()
            while line:
                offsets.append(mm.tell())
                line = mm.readline()
            mm.close()
        return offsets[:-1]  # Remove the last offset which is EOF
    
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
    
    def _process_chunk(self, args):
        """Process a chunk of text from a file based on line offsets"""
        file_path, offsets, tokenizer_model_path, lang, is_test = args
        
        # Load tokenizer in each worker process
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_model_path)
        
        sequences = []
        current_tokens = np.array([], dtype=np.uint16)
        
        with open(file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            for offset in offsets:
                mm.seek(offset)
                line = mm.readline().decode('utf-8', errors='ignore').strip()
                
                # Tokenize line
                line_tokens = np.array(tokenizer.encode(line), dtype=np.uint16)
                current_tokens = np.append(current_tokens, line_tokens)
                
                # Create full sequences of length sequence_length
                while len(current_tokens) >= self.sequence_length:
                    sequence = current_tokens[:self.sequence_length]
                    current_tokens = current_tokens[self.sequence_length:]
                    sequences.append(sequence)
            
            mm.close()
        
        return sequences, lang, is_test
    
    def _save_sequences_to_file(self, sequences: List[np.ndarray], file_path: str):
        """Save sequences to a binary file"""
        with open(file_path, 'ab') as f:
            for seq in sequences:
                seq.tofile(f)
    
    def calculate_offsets(self):
        """Calculate line offsets for all files"""
        print("Calculating line offsets for all files...")
        for lang, file_path in tqdm(self.file_paths.items()):
            self.file_offsets[lang] = self._calculate_line_offsets(file_path)
            print(f"{lang}: {len(self.file_offsets[lang])} lines")
    
    def save_processed_data(self, output_dir: str, chunk_size: int = 1000):
        """
        Process all data and save to:
        - One combined train file for all languages
        - Separate test files for each language
        
        Args:
            output_dir: Directory to save processed files
            chunk_size: Number of lines to process in each worker chunk
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create train file
        train_file = os.path.join(output_dir, 'train.bin')
        open(train_file, 'wb').close()
        
        # Create test files for each language
        test_files = {}
        for lang in self.file_paths.keys():
            test_file = os.path.join(output_dir, f'{lang}_test.bin')
            test_files[lang] = test_file
            open(test_file, 'wb').close()
        
        # Calculate offsets if not done already
        if not self.file_offsets:
            self.calculate_offsets()
        
        # Save metadata
        metadata = {
            "sequence_length": self.sequence_length,
            "vocab_size": self.tokenizer.get_piece_size(),
            "dtype": "uint16",
            "languages": {},
            "train_test_ratio": 1.0 - self.test_ratio,
            "files": {
                "train": os.path.basename(train_file),
                "test": {lang: os.path.basename(file) for lang, file in test_files.items()}
            }
        }
        
        # Process each language
        train_sequence_count = 0
        test_sequence_counts = {lang: 0 for lang in self.file_paths.keys()}
        
        for lang, file_path in tqdm(self.file_paths.items(), desc="Processing languages"):
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
                train_chunks.append((file_path, chunk_offsets, self.tokenizer_model_path, lang, False))
            
            # Create processing chunks (test)
            test_chunks = []
            for i in range(0, len(test_indices), chunk_size):
                chunk_indices = test_indices[i:i+chunk_size]
                chunk_offsets = [self.file_offsets[lang][idx] for idx in chunk_indices]
                test_chunks.append((file_path, chunk_offsets, self.tokenizer_model_path, lang, True))
            
            # Process all chunks with multiprocessing
            all_chunks = train_chunks + test_chunks
            total_chunks = len(all_chunks)
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(total_chunks, self.num_workers * 2)
            
            for i in range(0, total_chunks, batch_size):
                current_chunks = all_chunks[i:i+batch_size]
                with mp.Pool(processes=self.num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(self._process_chunk, current_chunks),
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
                
                # Force garbage collection to free memory
                results = None
        
        # Update metadata with sequence counts
        metadata["sequence_counts"] = {
            "train": train_sequence_count,
            "test": test_sequence_counts
        }
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSaved processed data to {output_dir}")
        print(f"Train file: {train_file} ({train_sequence_count} sequences)")
        for lang, count in test_sequence_counts.items():
            print(f"{lang} test file: {test_files[lang]} ({count} sequences)")

class FastBinaryDataLoader:
    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 32,
        shuffle: bool = True,
        sequence_length: int = 512,
        buffer_size: int = 100_000,  # Number of sequences to load into memory buffer
        prefetch: bool = True
    ):
        """
        Initialize the fast binary data loader with buffering
        
        Args:
            data_path: Path to the binary data file
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle the data
            sequence_length: Length of each sequence
            buffer_size: Number of sequences to buffer in memory
            prefetch: Whether to prefetch the next buffer in background
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        
        # Get file size to determine number of sequences
        file_size = os.path.getsize(data_path)
        self.n_sequences = file_size // (self.sequence_length * 2)  # uint16 = 2 bytes
        self.bytes_per_sequence = self.sequence_length * 2
        
        # Initialize state
        self.current_index = 0
        self.buffer = None
        self.buffer_start_idx = 0
        self.buffer_end_idx = 0
        self.current_buffer_idx = 0
        
        # For prefetching
        self.prefetch_thread = None
        self.next_buffer = None
        self.next_buffer_indices = None
        self.next_buffer_start_idx = 0
        self.next_buffer_end_idx = 0
        
        # Calculate number of buffers needed
        self.n_buffers = (self.n_sequences + self.buffer_size - 1) // self.buffer_size
        
        # Create shuffled buffer access order
        self.buffer_order = list(range(self.n_buffers))
        if self.shuffle:
            random.shuffle(self.buffer_order)
    
    def _load_buffer(self, buffer_idx):
        """Load a buffer of sequences from the data file"""
        start_sequence = buffer_idx * self.buffer_size
        end_sequence = min(start_sequence + self.buffer_size, self.n_sequences)
        sequences_to_load = end_sequence - start_sequence
        
        with open(self.data_path, 'rb') as f:
            # Seek to the start of the buffer
            f.seek(start_sequence * self.bytes_per_sequence)
            # Read the entire buffer at once
            raw_data = f.read(sequences_to_load * self.bytes_per_sequence)
            # Convert to numpy array
            data = np.frombuffer(raw_data, dtype=np.uint16)
            # Reshape to sequences
            buffer = data.reshape(sequences_to_load, self.sequence_length)
        
        buffer_indices = np.arange(sequences_to_load)
        if self.shuffle:
            np.random.shuffle(buffer_indices)
        
        return buffer, buffer_indices, start_sequence, end_sequence
    
    def _prefetch_next_buffer(self):
        """Prefetch the next buffer in background"""
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
        
        current_buffer_position = self.buffer_order.index(self.current_buffer_idx)
        if current_buffer_position < len(self.buffer_order) - 1:
            next_buffer_idx = self.buffer_order[current_buffer_position + 1]
            
            def load_buffer_thread():
                self.next_buffer, self.next_buffer_indices, self.next_buffer_start_idx, self.next_buffer_end_idx = self._load_buffer(next_buffer_idx)
            
            self.prefetch_thread = threading.Thread(target=load_buffer_thread)
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()
    
    def __len__(self):
        return self.n_sequences // self.batch_size
    
    def __iter__(self):
        # Reset state
        self.current_index = 0
        
        # Shuffle buffer order if needed
        if self.shuffle:
            random.shuffle(self.buffer_order)
        
        # Load first buffer
        self.current_buffer_idx = self.buffer_order[0]
        self.buffer, self.buffer_indices, self.buffer_start_idx, self.buffer_end_idx = self._load_buffer(self.current_buffer_idx)
        
        # Prefetch next buffer if needed
        if self.prefetch and len(self.buffer_order) > 1:
            self._prefetch_next_buffer()
        
        return self
    
    def __next__(self):
        if self.current_index >= self.n_sequences:
            raise StopIteration
        
        # Check if we need to load the next buffer
        if self.current_index >= self.buffer_end_idx:
            current_buffer_position = self.buffer_order.index(self.current_buffer_idx)
            
            if current_buffer_position >= len(self.buffer_order) - 1:
                # No more buffers
                raise StopIteration
            
            # Move to next buffer
            self.current_buffer_idx = self.buffer_order[current_buffer_position + 1]
            
            if self.prefetch and self.next_buffer is not None:
                # Use prefetched buffer
                self.buffer = self.next_buffer
                self.buffer_indices = self.next_buffer_indices
                self.buffer_start_idx = self.next_buffer_start_idx
                self.buffer_end_idx = self.next_buffer_end_idx
                self.next_buffer = None
                
                # Prefetch next buffer
                if current_buffer_position < len(self.buffer_order) - 2:
                    self._prefetch_next_buffer()
            else:
                # Load buffer synchronously
                self.buffer, self.buffer_indices, self.buffer_start_idx, self.buffer_end_idx = self._load_buffer(self.current_buffer_idx)
        
        # Prepare batch
        remaining_in_buffer = self.buffer_end_idx - self.current_index
        batch_size = min(self.batch_size, remaining_in_buffer)
        
        # Get batch indices relative to the buffer
        buffer_relative_indices = self.buffer_indices[(self.current_index - self.buffer_start_idx):(self.current_index - self.buffer_start_idx + batch_size)]
        
        # Get batch data
        batch = self.buffer[buffer_relative_indices]
        
        self.current_index += batch_size
        
        return batch

# Simple function to shuffle the training data after all languages are processed
def shuffle_binary_file(file_path, sequence_length=512, buffer_size=10000, output_path=None):
    """
    Shuffle the sequences in a binary file.
    Uses an external merge sort approach for memory efficiency.
    
    Args:
        file_path: Path to the binary file to shuffle
        sequence_length: Length of each sequence
        buffer_size: Number of sequences to load at once
        output_path: Path to save shuffled data (if None, overwrites original)
    """
    if output_path is None:
        output_path = file_path + ".shuffled"
    
    # Get file size
    file_size = os.path.getsize(file_path)
    n_sequences = file_size // (sequence_length * 2)  # uint16 = 2 bytes
    bytes_per_sequence = sequence_length * 2
    
    print(f"Shuffling {n_sequences} sequences...")
    
    # Create shuffled order of all sequences
    order = np.arange(n_sequences)
    np.random.shuffle(order)
    
    # Process in chunks to avoid memory issues
    with open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        for i in tqdm(range(0, n_sequences, buffer_size), desc="Shuffling chunks"):
            # Determine chunk size
            chunk_size = min(buffer_size, n_sequences - i)
            
            # Allocate buffer for this chunk
            buffer = np.zeros((chunk_size, sequence_length), dtype=np.uint16)
            
            # Read sequences in shuffled order for this chunk
            for j in range(chunk_size):
                seq_idx = order[i + j]
                f_in.seek(seq_idx * bytes_per_sequence)
                data = np.fromfile(f_in, dtype=np.uint16, count=sequence_length)
                buffer[j] = data
            
            # Write shuffled chunk
            buffer.tofile(f_out)
    
    # If we're replacing the original file, rename the shuffled file
    if output_path != file_path:
        os.rename(output_path, file_path)
    
    print(f"Shuffled data saved to {file_path}")

# Example usage:
if __name__ == "__main__":
    # Example configuration
    data_dir = "../txt_data"
    model_path = "./indic_tokenizer_100M/indic_tokenizer_100M.model"
    output_dir = "../bin_data"
    
    # Initialize and process the dataset
    dataset = MultilingualDatasetOptimized(
        data_dir=data_dir,
        model_path=model_path,
        test_ratio=0.05,
        sequence_length=1024
    )
    
    # Process and save all data to binary files
    dataset.save_processed_data(output_dir, chunk_size=500)
    
    # Shuffle the combined training file for better training
    print("\nShuffling the combined training file...")
    shuffle_binary_file(
        file_path=os.path.join(output_dir, 'train.bin'),
        sequence_length=512,
        buffer_size=10000
    )
    
    print("\nData processing complete!")
    print("To load the training data:")
    print("train_loader = FastBinaryDataLoader(")
    print(f"    data_path='{os.path.join(output_dir, 'train.bin')}',")
    print("    batch_size=32,")
    print("    shuffle=True,")
    print("    sequence_length=512,")
    print("    buffer_size=10000  # Adjust based on available RAM")
    print(")")
    
    print("\nTo evaluate on a specific language:")
    for lang in dataset.file_paths.keys():
        print(f"\n# For {lang} test data:")
        print(f"{lang}_test_loader = FastBinaryDataLoader(")
        print(f"    data_path='{os.path.join(output_dir, f'{lang}_test.bin')}',")
        print("    batch_size=32,")
        print("    shuffle=False,  # Usually no need to shuffle test data")
        print("    sequence_length=512,")
        print("    buffer_size=5000")
        print(")")
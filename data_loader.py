import os
import random
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set


class MultilingualBinaryDataLoader:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        shuffle: bool = True,
        sequence_length: int = 1024,
        buffer_size: int = 250_000,
        prefetch: bool = True,
        mode: str = "train"
    ):
        """
        Initialize a data loader for multiple binary language files.
        
        Args:
            data_dir: Directory containing binary language files
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            sequence_length: Length of each sequence in tokens
            buffer_size: Size of buffer for efficient loading
            prefetch: Whether to prefetch next buffer
            mode: 'train' or 'test'
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.mode = mode
        
        # Store loaders for both train and test modes
        self.train_loaders = {}
        self.test_loaders = {}
        
        # Store state for both modes
        self.train_weights = {}
        self.test_weights = {}
        self.train_total_sequences = 0
        self.test_total_sequences = 0
        
        # Current state tracking
        self.current_epoch = 0
        self.current_batch = 0
        
        # Initialize both train and test loaders
        self._initialize_loaders()
        
    def _initialize_loaders(self):
        """Initialize loaders for both train and test modes."""
        # Initialize train loaders
        train_files = self._get_language_files("train")
        for language, file_path in train_files.items():
            loader = FastBinaryDataLoader(
                data_path=file_path,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                sequence_length=self.sequence_length,
                buffer_size=self.buffer_size,
                prefetch=False
            )
            self.train_loaders[language] = loader
            self.train_total_sequences += loader.n_sequences
        
        # Calculate train weights
        for language, loader in self.train_loaders.items():
            self.train_weights[language] = loader.n_sequences / self.train_total_sequences
            
        # Initialize test loaders
        test_files = self._get_language_files("test")
        for language, file_path in test_files.items():
            loader = FastBinaryDataLoader(
                data_path=file_path,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                sequence_length=self.sequence_length,
                buffer_size=self.buffer_size,
                prefetch=False
            )
            self.test_loaders[language] = loader
            self.test_total_sequences += loader.n_sequences
            
        # Calculate test weights
        for language, loader in self.test_loaders.items():
            self.test_weights[language] = loader.n_sequences / self.test_total_sequences
            
    def _get_language_files(self, mode: str) -> Dict[str, str]:
        """Get all relevant language files based on the specified mode."""
        files = {}
        suffix = f"_{mode}.bin"
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(suffix):
                language = file_name.split("_")[0]
                files[language] = os.path.join(self.data_dir, file_name)
                
        return files
    
    @property
    def loaders(self):
        """Return current active loaders based on mode."""
        return self.train_loaders if self.mode == "train" else self.test_loaders
    
    @property
    def language_weights(self):
        """Return current active weights based on mode."""
        return self.train_weights if self.mode == "train" else self.test_weights
    
    @property
    def total_sequences(self):
        """Return total sequences in current mode."""
        return self.train_total_sequences if self.mode == "train" else self.test_total_sequences
    
    def get_languages(self) -> List[str]:
        """Return list of available languages in current mode."""
        return list(self.loaders.keys())
    
    def switch_mode(self, mode: str):
        """
        Switch between train and test modes without losing position.
        All loader states are preserved between mode switches.
        """
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'")
            
        if mode != self.mode:
            self.mode = mode
            # No need to reinitialize - just switch mode
    
    def __len__(self):
        """Return approximate number of batches in an epoch for current mode."""
        return self.total_sequences // self.batch_size
    
    def __iter__(self):
        """Reset iterator state for current mode."""
        self.current_batch = 0
        
        # Reset all language loaders for current mode
        for loader in self.loaders.values():
            loader.__iter__()
        
        return self
    
    def __next__(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get next mixed batch with proportional sampling from all languages.
        
        Returns:
            Tuple containing:
                - Batch of sequences as numpy array (batch_size, sequence_length)
                - List of language labels corresponding to each sequence
        """
        if self.current_batch >= len(self):
            raise StopIteration
            
        # Choose languages for this batch based on weights
        languages = list(self.loaders.keys())
        weights = [self.language_weights[lang] for lang in languages]
        
        # Sample languages for each item in the batch
        batch_languages = np.random.choice(
            languages, 
            size=self.batch_size, 
            p=weights, 
            replace=True
        )
        
        # Count how many sequences needed from each language
        lang_counts = {}
        for lang in batch_languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
        # Collect sequences from each loader
        batch_data = []
        batch_langs = []
        
        for lang, count in lang_counts.items():
            loader = self.loaders[lang]
            
            # Get required number of sequences from this language
            for _ in range(count):
                try:
                    # Get single example (might be inefficient but ensures proper distribution)
                    seq = next(loader)
                    
                    # FastBinaryDataLoader returns batches, so we need first item
                    if seq.shape[0] > 0:
                        batch_data.append(seq[0])
                        batch_langs.append(lang)
                except StopIteration:
                    # If we've exhausted this language, restart its iterator
                    loader.__iter__()
                    seq = next(loader)
                    if seq.shape[0] > 0:
                        batch_data.append(seq[0])
                        batch_langs.append(lang)
        
        # Ensure we have exactly batch_size sequences
        if len(batch_data) < self.batch_size:
            # If we couldn't get enough sequences, just duplicate some
            indices = list(range(len(batch_data)))
            if indices:  # Check if indices is not empty
                extra_indices = np.random.choice(indices, self.batch_size - len(batch_data))
                for idx in extra_indices:
                    batch_data.append(batch_data[idx])
                    batch_langs.append(batch_langs[idx])
        
        self.current_batch += 1
        
        return np.array(batch_data), batch_langs
    
    def get_language_batch(self, language: str) -> np.ndarray:
        """
        Get a batch from a specific language in current mode.
        
        Args:
            language: Language to get batch from
            
        Returns:
            Batch of sequences as numpy array
        """
        if language not in self.loaders:
            raise ValueError(f"Language '{language}' not available in current mode ({self.mode})")
            
        loader = self.loaders[language]
        try:
            return next(loader)
        except StopIteration:
            # Reset iterator and try again
            loader.__iter__()
            return next(loader)
            
    def get_test_batch(self, language: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Get a test batch, optionally for a specific language.
        This preserves the state of the current mode.
        
        Args:
            language: Optional language to get batch from
            
        Returns:
            If language specified: Batch of sequences as numpy array
            If language not specified: Tuple of (batch, language_labels)
        """
        current_mode = self.mode
        
        try:
            # Temporarily switch to test mode if needed
            if current_mode != "test":
                self.switch_mode("test")
                
            if language:
                return self.get_language_batch(language)
            else:
                return next(self)
        finally:
            # Switch back to original mode if needed
            if current_mode != "test":
                self.switch_mode(current_mode)
    
    def get_train_batch(self, language: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Get a training batch, optionally for a specific language.
        This preserves the state of the current mode.
        
        Args:
            language: Optional language to get batch from
            
        Returns:
            If language specified: Batch of sequences as numpy array
            If language not specified: Tuple of (batch, language_labels)
        """
        current_mode = self.mode
        
        try:
            # Temporarily switch to train mode if needed
            if current_mode != "train":
                self.switch_mode("train")
                
            if language:
                return self.get_language_batch(language)
            else:
                return next(self)
        finally:
            # Switch back to original mode if needed
            if current_mode != "train":
                self.switch_mode(current_mode)


# Keep the original FastBinaryDataLoader to use as a component
class FastBinaryDataLoader:
    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 64,
        shuffle: bool = True,
        sequence_length: int = 1024,
        buffer_size: int = 250_000,
        prefetch: bool = True
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        
        file_size = os.path.getsize(data_path)
        self.n_sequences = file_size // (self.sequence_length * 2)  # uint16 = 2 bytes
        self.bytes_per_sequence = self.sequence_length * 2

        self.current_index = 0
        self.buffer = None
        self.buffer_start_idx = 0
        self.buffer_end_idx = 0
        self.current_buffer_idx = 0

        self.prefetch_thread = None
        self.next_buffer = None
        self.next_buffer_indices = None
        self.next_buffer_start_idx = 0
        self.next_buffer_end_idx = 0
        self.prefetch_lock = threading.Lock()  # Ensures thread safety
        
        self.n_buffers = (self.n_sequences + self.buffer_size - 1) // self.buffer_size

        self.buffer_order = list(range(self.n_buffers))
        if self.shuffle:
            random.shuffle(self.buffer_order)

    def _load_buffer(self, buffer_idx):
        """Load a buffer of sequences from the data file"""
        start_sequence = buffer_idx * self.buffer_size
        end_sequence = min(start_sequence + self.buffer_size, self.n_sequences)
        sequences_to_load = end_sequence - start_sequence
        
        with open(self.data_path, 'rb') as f:
            f.seek(start_sequence * self.bytes_per_sequence)
            raw_data = f.read(sequences_to_load * self.bytes_per_sequence)
            data = np.frombuffer(raw_data, dtype=np.uint16)
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
                with self.prefetch_lock:
                    self.next_buffer, self.next_buffer_indices, self.next_buffer_start_idx, self.next_buffer_end_idx = self._load_buffer(next_buffer_idx)

            self.prefetch_thread = threading.Thread(target=load_buffer_thread)
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()

    def __len__(self):
        return self.n_sequences // self.batch_size

    def __iter__(self):
        self.current_index = 0

        if self.shuffle:
            random.shuffle(self.buffer_order)

        self.current_buffer_idx = self.buffer_order[0]
        self.buffer, self.buffer_indices, self.buffer_start_idx, self.buffer_end_idx = self._load_buffer(self.current_buffer_idx)

        if self.prefetch and len(self.buffer_order) > 1:
            self._prefetch_next_buffer()

        return self

    def __next__(self):
        if self.current_index >= self.n_sequences:
            raise StopIteration

        if self.current_index >= self.buffer_end_idx:
            current_buffer_position = self.buffer_order.index(self.current_buffer_idx)

            if current_buffer_position >= len(self.buffer_order) - 1:
                raise StopIteration

            self.current_buffer_idx = self.buffer_order[current_buffer_position + 1]

            if self.prefetch and self.next_buffer is not None:
                with self.prefetch_lock:
                    self.buffer = self.next_buffer
                    self.buffer_indices = self.next_buffer_indices
                    self.buffer_start_idx = self.next_buffer_start_idx
                    self.buffer_end_idx = self.next_buffer_end_idx
                    self.next_buffer = None

                if current_buffer_position < len(self.buffer_order) - 2:
                    self._prefetch_next_buffer()
            else:
                self.buffer, self.buffer_indices, self.buffer_start_idx, self.buffer_end_idx = self._load_buffer(self.current_buffer_idx)

        remaining_in_buffer = self.buffer_end_idx - self.current_index
        batch_size = min(self.batch_size, remaining_in_buffer)

        buffer_relative_start = self.current_index - self.buffer_start_idx
        buffer_relative_end = buffer_relative_start + batch_size
        buffer_relative_indices = self.buffer_indices[buffer_relative_start:buffer_relative_end]

        batch = self.buffer[buffer_relative_indices]

        self.current_index += batch_size

        return batch
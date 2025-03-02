import os
import random
import threading
import numpy as np

class FastBinaryDataLoader:
    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 64,  # Increased for n2-standard-8
        shuffle: bool = True,
        sequence_length: int = 1024,  # Updated to 1024
        buffer_size: int = 250_000,  # Increased for 32GB memory
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

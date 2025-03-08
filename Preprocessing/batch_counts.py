    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 64,  # Increased for n2-standard-8
        shuffle: bool = True,
        sequence_length: int = 1024,  # Updated to 1024
        buffer_size: int = 250_000,  # Increased for 32GB memory
        prefetch: bool = True
    )

from data_loader import 

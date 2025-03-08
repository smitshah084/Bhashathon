# Initialize the loader
from data_loader import MultilingualBinaryDataLoader

# Initialize with data directory
loader = MultilingualBinaryDataLoader(
    data_dir='bin_data',
    batch_size=64,
    sequence_length=1024,
    mode="train"  # Start in training mode
)

# Get a mixed batch with proportional sampling
batch, languages = next(loader)
print(f"Batch shape: {batch.shape}, Languages: {languages[:5]}...")

# Get a batch from a specific language
hindi_batch = loader.get_language_batch("hindi")
print(f"Hindi batch shape: {hindi_batch.shape}")

# Get test batch without changing mode
test_batch, test_langs = loader.get_test_batch()
print(f"Test batch shape: {test_batch.shape}")

# Get test batch for specific language
kannada_test = loader.get_test_batch("kannada")
print(f"Kannada test batch shape: {kannada_test.shape}")

# Switch to test mode permanently if needed
loader.switch_mode("test")

# Get 5 batches in train mode
for i in range(5):
    batch, languages = next(loader)
    print(f"Train batch {i+1}")

# Switch to test mode and get some batches
loader.switch_mode("test")
for i in range(3):
    batch, languages = next(loader)
    print(f"Test batch {i+1}")

# Switch back to train mode - will continue from batch 6
loader.switch_mode("train")
batch, languages = next(loader)
print("Resumed train batch 6")

# Get specific language batches without disrupting the counters
hindi_batch = loader.get_language_batch("hindi")
gujarati_test = loader.get_test_batch("gujarati")  # Doesn't change the mode

# Continue with train batches from where we left off
batch, languages = next(loader)
print("Still on train mode, now at batch 7")
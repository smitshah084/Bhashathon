from Preprocessing.data_loader import FastBinaryDataLoader

# train_bin_path = 'bin_data/train.bin'
# test_bin_paths = ['./bin_data/Hindi_test.bin',
#  './bin_data/gujrati_test.bin',
#  './bin_data/kannada_test.bin',
#  './bin_data/mar_test.bin',
#  './bin_data/mlym_test.bin',
#  './bin_data/orya_test.bin',]

paths = ['gujarati_processed_test.bin',   'kannada_processed_test.bin'     ,'marathi_processed_test.bin'   ,'odia_processed_train.bin',
'gujarati_processed_train.bin',  'kannada_processed_train.bin',    'marathi_processed_train.bin',
'hindi_processed_test.bin',      'malayalam_processed_test.bin'  , 'hindi_processed_train.bin' ,    'malayalam_processed_train.bin' , 'odia_processed_test.bin']

langs = ["Hin","Guj","Kann","Mar","mlym","Orya"]
total_tokens = 0
for path in paths:
        print(path,end=" ")
        path = './bin_data/' + path
        test_loader = FastBinaryDataLoader(
        data_path=path, 
        batch_size=8, 
        shuffle=False,  # No need to shuffle test data
        sequence_length=1024, 
        buffer_size=250_000,
        prefetch=False  # No need to prefetch in test
        )
        
        num_test_batches = len(test_loader)
        total_tokens += num_test_batches * 8 * 1024
        print(": ",num_test_batches)
        
print(total_tokens)
import sys;
sys.exit()

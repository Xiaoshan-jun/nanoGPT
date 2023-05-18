"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import tiktoken
file_names = [f for f in os.listdir('data/riskmap/') if f.startswith("train")]
combined_file_name = "data/riskmap/atraindata.txt"

# Open the combined file for writing
with open(combined_file_name, "w") as outfile:

    # Loop over the list of file names
    for file_name in file_names:
        file_name = os.path.join(os.path.dirname(__file__), file_name)
        # Open the current file for reading
        with open(file_name, "r") as infile:

            # Read the contents of the current file
            contents = infile.read()

            # Write the contents to the combined file
            outfile.write(contents)
file_names = [f for f in os.listdir('data/riskmap/') if f.startswith("val")]
combined_file_name = "data/riskmap/avaldata.txt"

# Open the combined file for writing
with open(combined_file_name, "w") as outfile:

    # Loop over the list of file names
    for file_name in file_names:
        file_name = os.path.join(os.path.dirname(__file__), file_name)
        # Open the current file for reading
        with open(file_name, "r") as infile:

            # Read the contents of the current file
            contents = infile.read()

            # Write the contents to the combined file
            outfile.write(contents)
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'atraindata.txt')
input_file_path2 = os.path.join(os.path.dirname(__file__), 'avaldata.txt')
with open(input_file_path, 'r') as f:
    train_data = f.read()
#train_data = train_data.split('\n')
print(f"length of dataset in characters: {len(train_data):,}")
with open(input_file_path2, 'r') as f:
    val_data = f.read()
#val_data = val_data.split('\n')



# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))




# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

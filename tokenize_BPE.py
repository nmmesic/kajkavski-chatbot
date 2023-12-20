import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

class BPE_token(object):
    def __init__(self):
        # Include Croatian alphabet characters in the ByteLevel pre-tokenizer
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel(added_tokens=["č", "ć", "ž", "š", "đ"])

    def train_bpe(self, paths, vocab_size=5000):
        # Adjust the vocab_size as needed
        trainer = BpeTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"], vocab_size=vocab_size)
        self.tokenizer.train(files=paths, trainer=trainer)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)

# Example Usage:
bpe_tokenizer = BPE_token()

# Replace 'path/to/dataset' with the path to your dataset (a directory containing text files).
dataset_path = "./dataset/dataset.txt"
dataset_files = [dataset_path]

# Train the BPE tokenizer with the Croatian alphabet
bpe_tokenizer.train_bpe(dataset_files, vocab_size=10000)  # Adjust vocab_size as needed

save_location = './tokenized_data/'
bpe_tokenizer.save_tokenizer(save_location, prefix='bpe_tokens')
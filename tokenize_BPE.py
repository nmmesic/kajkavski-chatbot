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
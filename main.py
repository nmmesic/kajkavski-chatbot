import os
from tokenize_BPE import BPE_token
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path

block_size = 100
BATCH_SIZE = 16
BUFFER_SIZE = 1000

DATASET_PATH = Path('/home/jnovosel/kajkavski-chatbot/dataset/dataset.txt')
TOKENIZED_PATH = Path('/home/jnovosel/kajkavski-chatbot/tokenized_data')
OUTPUT_DIR_PATH = Path('/home/jnovosel/kajkavski-chatbot/model_bn_custom')


def tokenizeData(tokenizer, paths):
    single_string = ''
    if len(paths) > 1:
        for filename in paths:
            with open(filename, "r", encoding='utf-8') as f:
                x = f.read()
            single_string += x + tokenizer.eos_token
    else:
        with open(paths[0], "r", encoding='utf-8') as f:
            x = f.read()
        single_string += x + tokenizer.eos_token
    string_tokenized = tokenizer.encode(single_string)
    return string_tokenized

def initializeTokenizer(dataset_path, save_location):
    bpe_tokenizer = BPE_token()

    dataset_files = [dataset_path]

    # Train the BPE tokenizer with the Croatian alphabet
    bpe_tokenizer.train_bpe(dataset_files, vocab_size=10000)  # Adjust vocab_size as needed

    bpe_tokenizer.save_tokenizer(save_location)

def initializeModel(save_path):
    # loading tokenizer from the saved model path
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>" })
    # creating the configurations from which the model can be made
    config = GPT2Config(vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    # creating the model
    model = TFGPT2LMHeadModel(config)
    return (model, tokenizer)

def train(model, string_tokenized):

    # make dataset
    examples = []
    for i in range(0, len(string_tokenized) - block_size + 1, block_size):
        examples.append(string_tokenized[i:i + block_size])

    inputs, labels = [], []
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # defining our optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    # definining our loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # defining our metric which we want to observe
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # compiling the model
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

    num_epoch = 10
    print("am here")
    history = model.fit(dataset, epochs=num_epoch)

def save(model, tokenizer, output_dir = OUTPUT_DIR_PATH):
    # creating directory if it is not present

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    model_to_save = model.module if hasattr(model, 'module') else model

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    # save model and model configs
    model.save_pretrained(output_dir)
    model_to_save.config.to_json_file(output_config_file)
    # save tokenizer
    tokenizer.save_pretrained(output_dir)

def main():
    initializeTokenizer(str(DATASET_PATH), str(TOKENIZED_PATH))

    model, tokenizer = initializeModel(str(TOKENIZED_PATH))

    string_tokenized = tokenizeData(tokenizer, [str(DATASET_PATH)])

    train(model, string_tokenized)

    save(model, tokenizer)    

if __name__ == "__main__":
    main()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


OUTPUT_DIR_PATH = 'model_bn_custom_noah'

with tf.device('GPU:1'):
  tokenizer = GPT2Tokenizer.from_pretrained(str(OUTPUT_DIR_PATH))
  model = TFGPT2LMHeadModel.from_pretrained(str(OUTPUT_DIR_PATH))

  
  while True:
    text = input('>') # encoding the input text
    input_ids = tokenizer.encode(text, return_tensors='tf')# getting out output
    beam_output = model.generate(
      input_ids,
      max_length = 50,
      num_beams = 5,
      temperature = 0.7,
      no_repeat_ngram_size=2,
      num_return_sequences=5,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id
    )

    # print(tokenizer.decode(beam_output[0]).split('.')[0] + '.')
    output = tokenizer.decode(beam_output[0])[len(text):].split('.')
    print(output[0] + '. ' + output[1] + '.')

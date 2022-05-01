import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import tensorflow as tf
from tensorflow import keras
import models
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from main import *
import utils
import pandas as pd

VOCAB_SIZE = 8192
MAX_SAMPLES = 50000
BUFFER_SIZE = 20000
MAX_LENGTH = 40
EMBED_DIM = 256
LATENT_DIM = 512
NUM_HEADS = 8
BATCH_SIZE = 64

def init_vectorizer(path="./data/dailydialog/all_data.csv"):
    df = pd.read_csv(path)
    vectorizer = layers.TextVectorization(
            8192,
            standardize=utils.preprocess_text,
            output_mode="int",
            output_sequence_length=40,
        )
    input_texts, output_texts = utils.vectorize_data(df)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices((input_texts + output_texts)).batch(128))
    return vectorizer


def decode_sentence(input_sentence):
    # Mapping the input sentence to tokens and adding start and end tokens
    tokenized_input_sentence = vectorizer(
        tf.constant("[start] " + utils.preprocess_text(input_sentence) + " [end]")
    )
    # Initializing the initial sentence consisting of only the start token.
    tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
    decoded_sentence = ""

    for i in range(MAX_LENGTH):
        # Get the predictions
        predictions = fnet.predict(
            {
                "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
                "decoder_inputs": tf.expand_dims(
                    tf.pad(
                        tokenized_target_sentence,
                        [[0, MAX_LENGTH - tf.shape(tokenized_target_sentence)[0]]],
                    ),
                    0,
                ),
            }
        )
        # Calculating the token with maximum probability and getting the corresponding word
        sampled_token_index = tf.argmax(predictions[0, i, :])
        sampled_token = VOCAB[sampled_token_index.numpy()]
        # If sampled token is the end token then stop generating and return the sentence
        if tf.equal(sampled_token_index, VOCAB.index("[end]")):
            break
        decoded_sentence += sampled_token + " "
        tokenized_target_sentence = tf.concat(
            [tokenized_target_sentence, [sampled_token_index]], 0
        )

    return decoded_sentence


if __name__ == "__main__":
    print("Welcome to the conversational AI")
    fnet = load_model('fnet_model.h5', custom_objects={"PositionalEmbedding": models.PositionalEmbedding, "FNetEncoder": models.FNetEncoder, "FNetDecoder": models.FNetDecoder})
    vectorizer = init_vectorizer()
    VOCAB = vectorizer.get_vocabulary()
    
    while True:
        text = input()
        if text == "q" or text == "quit":
            break
        response = decode_sentence(text)
        print(response)


import pandas as pd
import os
import json
import re
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import tensorflow_datasets as tfds
import sys
import random
import pickle

def format_dataset(tokenized_inputs, tokenized_outputs):
    return (
        {
            'enc_inputs': tokenized_inputs,
            'dec_inputs': tokenized_outputs[:-1]
        },
        {
            'outputs': tokenized_outputs[1:]
        },
    )

def vectorize_text(inputs, outputs):
    inputs, outputs = vectorizer(inputs), vectorizer(outputs)
    # One extra padding token to the right to match the output shape
    outputs = tf.pad(outputs, [[0, 1]])
    return (
        {"encoder_inputs": inputs, "decoder_inputs": outputs[:-1]},
        {"outputs": outputs[1:]},
    )

def preprocess_text(sentence):
    sentence = tf.strings.lower(sentence)
    # Adding a space between the punctuation and the last word to allow better tokenization
    sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")
    # Replacing multiple continuous spaces with a single space
    sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
    # Replacing non english words with spaces
    sentence = tf.strings.regex_replace(sentence, r"[^a-z?.!,]+", " ")
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence
    
vectorizer = layers.TextVectorization(
        8192,
        standardize=preprocess_text,
        output_mode="int",
        output_sequence_length=40,
    )


def make_datasets(df, batch_size=64, VOCAB_SIZE=8192, BUFFER_SIZE=20000):
    input_texts, output_texts = vectorize_data(df)
    dataset = tf.data.Dataset.from_tensor_slices((input_texts, output_texts))
    dataset = dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def load_data(path="./data/dailydialog/all_data.csv", batch_size=64):
    df = pd.read_csv(path)
    num_conversations = max(df["conversation_id"])
    test_conversation_ids = random.sample(range(1, num_conversations + 1), int(num_conversations*0.15))
    input_texts, output_texts = vectorize_data(df)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices((input_texts + output_texts)).batch(128))
    val_df = df[df["conversation_id"].isin(test_conversation_ids)]
    train_df = df[~df["conversation_id"].isin(test_conversation_ids)]
    train_ds = make_datasets(train_df)
    val_ds = make_datasets(val_df)
    return train_ds, val_ds, 15000

def vectorize_data(df, num_words=None, max_len=None):
    input_texts = []
    target_texts = []
    for index in tqdm(range(df.shape[0])):
        if index == 0:
            continue
        input_text = df.iloc[index - 1]
        target_text = df.iloc[index]
        if input_text.conversation_id == target_text.conversation_id:
            input_text = input_text.message
            target_text = target_text.message
            if input_text and target_text:
                input_texts.append(str(input_text))
                target_texts.append(str(target_text))
                
    return input_texts, target_texts

# def tokenize_texts(input_texts, target_texts):
#     tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(input_texts + target_texts, target_vocab_size=8000)

#     START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
#     MAX_SENTENCE_LENGTH = 40
#     VOCAB_SIZE = tokenizer.vocab_size + 2

#     tokenized_inputs, tokenized_outputs = [], []
  
#     for (sentence1, sentence2) in zip(input_texts, target_texts):
#         # tokenize sentence
#         sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
#         sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
#         # check tokenized sentence max length
#         if len(sentence1) <= MAX_SENTENCE_LENGTH and len(sentence2) <= MAX_SENTENCE_LENGTH:
#             tokenized_inputs.append(sentence1)
#             tokenized_outputs.append(sentence2)
    
#     # pad tokenized sentences
#     tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=MAX_SENTENCE_LENGTH, padding='post')
#     tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=MAX_SENTENCE_LENGTH, padding='post')
#     return tokenized_inputs, tokenized_outputs, VOCAB_SIZE


    
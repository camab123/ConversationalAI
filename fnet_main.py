import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils
import models
import math


if __name__ == "__main__":
    BATCH_SIZE = 64
    train_ds, val_ds, vocab_size = utils.load_data(batch_size=BATCH_SIZE)
    tf.keras.backend.clear_session()

    fnet = models.create_model()
    fnet.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    fnet.fit(train_ds, epochs=200, validation_data=val_ds)
    fnet.save("fnet_model.h5")
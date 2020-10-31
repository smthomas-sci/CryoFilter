"""
Contains the model to train / predict

Author: Simon Thomas
Date: 30th October 2020

Requirements (available by pip / conda):
- tensorflow

"""

from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model


def build_model(img_dim: int = 28, dropout=False):

    model_input_1 = Input(shape=(img_dim, img_dim, 1), name="image_in")
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(model_input_1)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    feat1 = Flatten()(x)

    model_input_2 = Input(shape=(img_dim, img_dim, 1), name="robert_in")
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(model_input_2)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    feat2 = Flatten()(x)

    x = Concatenate()([feat1, feat2])

    if dropout:
        x = Dropout(0.8)(x)

    model_output = Dense(1, activation="sigmoid")(x)

    return Model(inputs=[model_input_1, model_input_2],
                 outputs=[model_output],
                 name="classifier")


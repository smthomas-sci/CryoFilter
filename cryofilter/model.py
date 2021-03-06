"""
Contains the model to train / predict

Author: Simon Thomas
Date: 30th October 2020

Requirements (available by pip / conda):
- tensorflow

"""

from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_model(img_dim: int = 28, dropout=False, f=32):

    model_input_1 = Input(shape=(img_dim, img_dim, 1), name="image_in")
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(model_input_1)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    feat1 = Flatten()(x)

    model_input_2 = Input(shape=(img_dim, img_dim, 1), name="robert_in")
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(model_input_2)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    feat2 = Flatten()(x)

    feat_both = Concatenate(name="features")([feat1, feat2])

    x = feat_both

    if dropout:
        x = Dropout(0.8)(x)

    model_output = Dense(1, activation="sigmoid")(x)

    model_outputs = [model_output]

    return Model(inputs=[model_input_1, model_input_2],
                 outputs=model_outputs,
                 name="classifier")


def straight_through_model(img_dim, dropout=False, f=64):
    # Dummy used to simplify generator logic
    dummy_input = Input(shape=(img_dim, img_dim, 1), name="dummy_in")
    # real model
    model_input = Input(shape=(img_dim, img_dim, 1), name="image_in")
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(model_input)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2D(f, 3, activation="relu", strides=1, padding="same")(x)
    x = AveragePooling2D()(x)
    feat = Flatten(name="features")(x)

    x = feat

    if dropout:
        x = Dropout(dropout)(x)

    model_output = Dense(1, activation="sigmoid")(x)

    model_outputs = [model_output]

    return Model(inputs=[model_input, dummy_input],
                 outputs=model_outputs,
                 name="straight_classifier")


if __name__ == "__main__":

    import tensorflow as tf

    # # --------------------- DEBUG -------------------------------------- #
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # # ----------------------------------------------------------------- #

    model = straight_through_model(28)

    model.summary()

    out = model(tf.random.normal((1, 512, 512, 1)))

    print(out.shape)





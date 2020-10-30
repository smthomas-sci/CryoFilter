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

if __name__ == "__main__":

    import numpy as np
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


    model = build_model(28)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss="binary_crossentropy",
                  metrics=["acc"])


    #[print(thing.shape) for thing in model.inputs]

    X = [np.random.random((1, 28, 28, 1)), np.random.random((1, 28, 28, 1))]
    y = np.ones((1, 1))
    y_pred = model.predict(X)

    #out = model.train_on_batch(X, y)

    print(y, y_pred)





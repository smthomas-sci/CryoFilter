"""
Trains the model

Author: Simon Thomas
Date: 30th October 2020

Requirements (available by pip / conda):
- tensorflow

"""
import argparse
import os

import tensorflow as tf

from cryofilter.data import DataGenerator
from cryofilter.model import build_model


parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('--pos', type=str, nargs='+',
                    help='mrcs files belonging to the positive class')
parser.add_argument('--neg', type=str, nargs='+',
                    help='mrcs files belonging to the negative class')
parser.add_argument('--weight_dir', type=str,
                    help='Directory to saves weights in. Weights ~= 500kb')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of images to train at once - between 12 and 128 is fine')
parser.add_argument('--split', type=float, default=0.7,
                    help='Train : Validation split')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of times to train on the whole dataset')
parser.add_argument('--img_dim', type=int, default=28,
                    help='The size to resize to e.g. 256x256 -> 28x28. Larger is more computation')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate of network - default is pretty good')
parser.add_argument('--augmentation', type=bool, default=True,
                    help='Using data augmentation- default is True')
parser.add_argument('--history_dir', type=str, default=".",
                    help='Diretory to save history.csv')

args = parser.parse_args()

# # --------------------- DEBUG -------------------------------------- #
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # ----------------------------------------------------------------- #


LEARNING_RATE = args.learning_rate
IMG_DIM = args.img_dim
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
SPLIT = args.split
WEIGHT_DIR = args.weight_dir
HISTORY_DIR = args.history_dir
POS_FILES = args.pos
NEG_FILES = args.neg
AUGMENTATION = args.augmentation

# 1. Setup data
generator = DataGenerator(pos_files=POS_FILES,
                          neg_files=NEG_FILES,
                          batch_size=BATCH_SIZE,
                          split=SPLIT,
                          img_dim=IMG_DIM,
                          data_augmentation=AUGMENTATION
                          )

# 2. Build and prepare model for training
Acc = tf.keras.metrics.BinaryAccuracy()
Loss = tf.keras.losses.BinaryCrossentropy()

model = build_model(img_dim=IMG_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss="binary_crossentropy",
              metrics=[Acc])

# 3. Train model
steps_per_epoch = (generator.train.n // generator.batch_size) + 1
validation_steps = (generator.val.n // generator.batch_size) + 1

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    print("\nEpoch", epoch+1, "/", EPOCHS)
    train_progress = tf.keras.utils.Progbar(steps_per_epoch)

    # Train
    for step in range(steps_per_epoch):
        x1, x2, y = generator.train[step]
        loss, acc = model.train_on_batch([x1, x2], y)
        train_progress.update(step, values=[("loss", loss), ("acc", acc)])
    train_progress.update(step+1, finalize=True)

    # Validation
    val_progress = tf.keras.utils.Progbar(validation_steps)
    Acc.reset_states()
    print()
    for step in range(validation_steps):
        x1, x2, y = generator.val[step]
        y_pred = model.predict([x1, x2])

        Acc.update_state(y, y_pred)
        val_loss = Loss(y, y_pred).numpy()
        val_acc = Acc.result().numpy()

        val_progress.update(step, values=[("val_loss", val_loss), ("val_acc", val_acc)])
    val_progress.update(step+1, finalize=True)

    # Save results
    history["train_acc"].append(acc)
    history["train_loss"].append(loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Save model weights
    weight_path = os.path.join(WEIGHT_DIR, f"weights_epoch_{epoch}_val_acc_{val_acc:.3}.h5")
    model.save_weights(weight_path)
    print()

# EPOCH END
filename = os.path.join(HISTORY_DIR, "cryofilter/history.csv")
print("Saving history at:", filename)
print("-------------------- HISTORY -----------------")
with open(filename, "w") as f:
    line = ",".join(["-"] + list(history.keys())) + "\n"
    f.write(line)
    print(line)

    for step in range(EPOCHS):
        line = ",".join([str(step)] + [str(history[key][step]) for key in history.keys()]) + "\n"
        f.write(line)

        print(line)

generator.close()

# ------------ END ------------ #

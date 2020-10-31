"""

Predict and save scores for mrcs file

Author: Simon Thomas
Date: 31 October 2020

Note!
- Data augmentation is off.
- Shuffle is off.

"""

import argparse

import tensorflow as tf

from cryofilter.data import DataGenerator
from cryofilter.model import build_model
from cryofilter.utils import get_predictions

parser = argparse.ArgumentParser(description='Score mrc file')
parser.add_argument('--mrc', type=str,
                    help='mrcs files to score')
parser.add_argument('--weights', type=str,
                    help='Path to the model weights')
parser.add_argument('--out_file', type=str,
                    help='full path and file name (include ".csv") to save scores')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of images to train at once - between 12 and 128 is fine')
parser.add_argument('--img_dim', type=int, default=28,
                    help='The size to resize to e.g. 256x256 -> 28x28. Larger is more computation')
args = parser.parse_args()


# # --------------------- DEBUG -------------------------------------- #
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # ----------------------------------------------------------------- #

IMG_DIM = args.img_dim
BATCH_SIZE = args.batch_size
WEIGHTS = args.weights
OUT_FILE = args.out_file
MRC_FILE = args.mrc

# 1. Setup data
generator = DataGenerator(pos_files=[MRC_FILE],
                          neg_files=[],
                          batch_size=BATCH_SIZE,
                          split=1.,
                          img_dim=IMG_DIM,
                          data_augmentation=False,
                          shuffle=False
                          )

# 2. Prepare Model
model = build_model(IMG_DIM)
model.load_weights(WEIGHTS)

# 3. Get predictions
y_true, y_pred = get_predictions(model, generator)

# 4. Save Scores
with open(OUT_FILE, "w") as f:
    for i, p in enumerate(y_pred):
        line = str(i) + ", " + f"{p[0]:.3f}" + "\n"
        f.write(line)
print("Scores saved:", OUT_FILE)

generator.close()




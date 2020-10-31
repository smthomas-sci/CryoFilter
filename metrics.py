"""

Model performance summary.

Author: Simon Thomas
Date: 31 October 2020


Note! Data augmentation is off.

"""

import argparse

import tensorflow as tf

from cryofilter.data import DataGenerator
from cryofilter.model import build_model
from cryofilter.utils import get_predictions, create_roc_plot, compute_metrics


parser = argparse.ArgumentParser(description='Evaluate network')
parser.add_argument('--pos', type=str, nargs='+',
                    help='mrcs files belonging to the positive class')
parser.add_argument('--neg', type=str, nargs='+',
                    help='mrcs files belonging to the negative class')
parser.add_argument('--weights', type=str,
                    help='Path to the model weights')
parser.add_argument('--out_dir', type=str,
                    help='Where to save the ROC plot.')
parser.add_argument('--split', type=float, default=0.7,
                    help='The split for training : validation. Use 1 for evaluate the whole dataset')
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
SPLIT = args.split
WEIGHTS = args.weights
OUT_DIR = args.out_dir
POS_FILES = args.pos
NEG_FILES = args.neg

# 1. Setup data
generator = DataGenerator(pos_files=POS_FILES,
                          neg_files=NEG_FILES,
                          batch_size=BATCH_SIZE,
                          split=SPLIT,
                          img_dim=IMG_DIM,
                          data_augmentation=False
                          )

# 2. Prepare Model
model = build_model(IMG_DIM)
model.load_weights(WEIGHTS)

# 3. Get predictions
y_true, y_pred = get_predictions(model, generator)

# 4. Compute metrics
metrics = compute_metrics(y_true, y_pred)

print()
print("--------------------------")
print("   Performance Summary    ")
print("--------------------------")

for key in metrics:
    if key != "ROC":
        if key == "CM":
            print(key, ":")
            print(metrics[key])
        else:
            print(key.capitalize(), ":", "{0:.3f}".format(metrics[key]))
print()

# 5. Create Plots
create_roc_plot(metrics, OUT_DIR)

generator.close()


# CryoFitler

**Author**: Simon Thomas

**Date**: October 2020

**Contact**: simon.thomas@uq.edu.au


# Training

The default parameters for training are reasonable and it may not be that necessary to tune the network.

```bash
train.py [-h] [--pos POS [POS ...]]
              [--neg NEG [NEG ...]]
              [--weight_dir WEIGHT_DIR]
              [--batch_size BATCH_SIZE default=32]
              [--split SPLIT default=0.7]
              [--epochs EPOCHS default=100]
              [--img_dim IMG_DIM default=28]
              [--learning_rate LEARNING_RATE default=0.001]
              [--history_dir HISTORY_DIR default="."]
```

For example, the following will start the training script:
```
python train.py --pos ./data/pos.mrcs ./data/pos_top.mrcs --neg ./data/neg.mrcs --weight_dir ./weights/ --batch_size 32 --split 0.7 --epochs 10 --img_dim 28 --learning_rate 0.001 --history_dir "."
```

To select the best model which minimise **over-fitting**, select the weights where the validation
accuracy is at its highest before it stops improving. This model can be used for prediction.


# Test

Testing and/or prediction in the wild is done using
 [Test Time Augmentation](https://www.nature.com/articles/s41598-020-61808-3) 
 to improve performance.
 
 This can be done using `python score.py [mrcfile]` with an output of a score between
 0-1 for each image, in the order found in the stack. The scores are saved as a
 csv file.
 
 **Note** - This has not been implemented yet.



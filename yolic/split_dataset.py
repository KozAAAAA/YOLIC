"""Split images into train, val, test sets"""

import splitfolders

NOT_SPLIT_DATA_DIR = "./data/images/"
SPLIT_DATA_DIR = "./data/split_images/"

if __name__ == "__main__":
    splitfolders.ratio(
        NOT_SPLIT_DATA_DIR, output=SPLIT_DATA_DIR, seed=1337, ratio=(0.8, 0.1, 0.1)
    )

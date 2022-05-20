import gc
import pathlib

import numpy as np
import pandas as pd
import torch

from dataset_utils import create_dataset, get_categories
from inception_v3_model_wrapper import InceptionV3ModelWrapper

# Train model if True. Load model from file otherwise.
TRAIN = False
DATA_ROOT_DIR = '../data/'
IMGS_DIR = 'cifar_10'

if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    categories_names = get_categories(DATA_ROOT_DIR, IMGS_DIR)
    model_wrapper = InceptionV3ModelWrapper(len(categories_names), device)
    dataset_for_submission = create_dataset(model_wrapper.get_expected_img_size(), DATA_ROOT_DIR, IMGS_DIR, False)

    if TRAIN:
        train_dataset, train_validation_dataset = create_dataset(model_wrapper.get_expected_img_size(), DATA_ROOT_DIR,
                                                                 IMGS_DIR)
        val_acc_history = model_wrapper.train(train_dataset, train_validation_dataset, True)
        submission_results, img_names_column = model_wrapper.predict(dataset_for_submission)
    else:
        submission_results, img_names_column = model_wrapper.predict(dataset_for_submission,
                                                                     './model_from_last_train.pth')

    # Write results into submission file
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    class_index_to_label = [
        "AUTOMOBILE",
        "DEER",
        "HORSE",
        "SHIP",
        "TRUCK",
        "FROG",
        "BIRD",
        "AIRPLANE",
        "DOG",
        "CAT"
    ]

    sample_submission['Category'] = [class_index_to_label[index] for index in
                                     np.concatenate(submission_results)]
    sample_submission['Id'] = [pathlib.Path(p).stem for p in np.concatenate(img_names_column)]

    sample_submission.to_csv('cifar_submission10.csv', index=False)

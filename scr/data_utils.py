import numpy as np
import random
import torch 
import dgl 
import os 

def set_seeds(seed):
    """
    Set seeds for reproducibility across all libraries. 
    Args:
        seed: Integer seed value 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)

def split_data(flat_data, non_flat_data, option=0):
    """
    option0: only have flat data to work with, create train/val/test sets out of that
    option1: only train and validate on flat data and test accuracy on non_flat data 
    option2: train and validate on both flat and non flat data and test accuracy on both data 
    Both options ensure equal amounts of flat/non-flat data in each split where applicable.
    """
    if option == 0:
        random.shuffle(flat_data)
        n = len(flat_data)
        # have a 70 / 15 / 15 split for training evaluation and test data 
        split1 = int(n * 0.9)

        training = flat_data[:split1]
        validation = flat_data[split1:]

    elif option == 1:
        random.shuffle(flat_data)
        n = len(flat_data)
    
        # use 0.7 and 0.3 for a 70/30 split  
        split1 = int(n * 0.7)
        training = flat_data[:split1]
        validation = flat_data[split1:]

        # test set is all non flat data
        test = non_flat_data

    elif option == 2:
        # Shuffle both datasets separately
        random.shuffle(flat_data)
        random.shuffle(non_flat_data)

        n_flat = len(flat_data)
        flat_split_1 = int(n_flat*0.7)
        flat_split_2 = int(n_flat*0.85)
        n_non_flat = len(non_flat_data)
        non_flat_split_1 = int(n_non_flat*0.7)
        non_flat_split_2 = int(n_non_flat*0.85)
        
        flat_train = flat_data[:flat_split_1]
        flat_val = flat_data[flat_split_1:flat_split_2]
        flat_test = flat_data[flat_split_2:]

        non_flat_train = non_flat_data[:non_flat_split_1]
        non_flat_val = non_flat_data[non_flat_split_1:non_flat_split_2]
        non_flat_test = non_flat_data[non_flat_split_2:]

        # Combine the datasets
        training = flat_train + non_flat_train
        validation = flat_val + non_flat_val
        test = flat_test + non_flat_test
        
        # Shuffle the combined datasets
        random.shuffle(training)
        random.shuffle(validation)
        random.shuffle(test)

    else:
        raise ValueError("Option must be either 1 or 2")


    # with open(f"../{sub_folder}/output_seed_{seed}.txt", "a") as file: 
    #     file.write(f"Split with seed {seed}: {len(training)} training, {len(validation)} validation, {len(test)} test\n")

    #     if option == 2:
    #         flat_count_train = sum(1 for item in training if item in flat_data)
    #         non_flat_count_train = len(training) - flat_count_train
    #         flat_count_val = sum(1 for item in validation if item in flat_data)
    #         non_flat_count_val = len(validation) - flat_count_val
    #         flat_count_test = sum(1 for item in test if item in flat_data)
    #         non_flat_count_test = len(test) - flat_count_test
            
    #         file.write(f"Balance - Training: {flat_count_train} flat, {non_flat_count_train} non-flat\n")
    #         file.write(f"Balance - Validation: {flat_count_val} flat, {non_flat_count_val} non-flat\n")
    #         file.write(f"Balance - Test: {flat_count_test} flat, {non_flat_count_test} non-flat\n")

    return (training, validation)

# Calculate confidence intervals for each metric
def get_confidence_interval(scores):
    scores = np.array(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1)
    
    # For normal approximation
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * std_dev
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, std_dev, ci_lower, ci_upper


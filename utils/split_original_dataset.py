import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.8):

    random.seed(42) #needed?

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        random.shuffle(files)

        split_index = int(len(files) * train_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for file_name in train_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(train_class_dir, file_name))

        for file_name in test_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(test_class_dir, file_name))

        print(f"Processed class '{class_name}': {len(train_files)} train, {len(test_files)} test")

# Example usage
input_directory = "../utils/test_parent/train"
output_directory = "../data/hiflatnet_training_data"
split_dataset(input_directory, output_directory)
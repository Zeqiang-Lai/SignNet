import os
import argparse
import random


def packing_test_dataset(root_path):
    # we only need to process test part
    test_dataset_path = os.path.join(root_path, 'asl_alphabet_test/asl_alphabet_test')
    for file in os.listdir(test_dataset_path):
        if file.startswith('.'):
            continue
        label = file.split('_')[0]
        print(label)

        sub_dir = os.path.join(test_dataset_path, label)
        os.makedirs(sub_dir, exist_ok=True)
        os.rename(os.path.join(test_dataset_path, file), os.path.join(sub_dir, file))


def split_train_val_dataset(root_path, shuffle=True, val_ratio=0.1):
    # WARNING: YOU SHOULD NOT RUN THIS FOR MULTIPLE TIME
    train_dataset_path = os.path.join(root_path, 'asl_alphabet_train/asl_alphabet_train')
    val_dataset_path = os.path.join(root_path, 'asl_alphabet_val/asl_alphabet_val')

    os.makedirs(val_dataset_path, exist_ok=True)

    for img_dir in os.listdir(train_dataset_path):
        if img_dir.startswith('.'):
            continue

        print("Process " + img_dir)
        img_dir_path = os.path.join(train_dataset_path, img_dir)

        imgs = os.listdir(img_dir_path)
        if shuffle:
            random.shuffle(imgs)

        bound = int(len(imgs) * (1 - val_ratio))
        val_imgs = imgs[bound:]

        # move images
        val_img_dir_path = os.path.join(val_dataset_path, img_dir)
        os.makedirs(val_img_dir_path, exist_ok=True)
        for img in val_imgs:
            img_path = os.path.join(img_dir_path, img)
            val_img_path = os.path.join(val_img_dir_path, img)
            os.rename(img_path, val_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASL Alphabet Dataset Preprocessor')
    parser.add_argument('--dataset-path', type=str, default='./data',
                        help='The root path of the dataset')
    args = parser.parse_args()

    split_train_val_dataset(args.dataset_path)

import os
import argparse

if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description='ASL Alphabet Dataset Preprocessor')  
    parser.add_argument('--dataset-path', type=str, default='./data',
                        help='The root path of the dataset')
    args = parser.parse_args()

    # we only need to process test part
    test_dataset_path = os.path.join(args.dataset_path, 'asl_alphabet_test/asl_alphabet_test')
    for file in os.listdir(test_dataset_path):
        if(file.startswith('.')):
            continue
        label = file.split('_')[0]
        print(label)

        sub_dir = os.path.join(test_dataset_path, label)
        os.makedirs(sub_dir, exist_ok=True)
        os.rename(os.path.join(test_dataset_path, file), os.path.join(sub_dir, file))


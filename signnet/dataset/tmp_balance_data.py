import argparse
import os
import re

from dlcommon.os import list_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--maxn', type=int, required=True)
    args = parser.parse_args()

    data_path = args.input
    maxn = args.maxn
    for dir in list_dir(data_path):
        print('Process ' + dir)
        count = 0
        files = list_dir(os.path.join(data_path, dir))
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for file in files:
            file_path = os.path.join(data_path, dir, file)
            if count < len(files) - maxn:
                os.remove(file_path)
                print(file_path)
            count += 1

import json


def save_dict(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)

import os
import argparse
import numpy as np

def write_to_file(file_name, data):
    with open(file_name, "w") as f:
        for s, r, o in data:
            f.write('\t'.join([s, r, o]) + '\n')


def main(params):
    with open(os.path.join(params.main_dir, 'data', params.dataset, 'train.txt')) as f:
        train_data = [line.split() for line in f.read().split('\n')[:-1]]
    with open(os.path.join(params.main_dir, 'data', params.dataset, 'valid.txt')) as f:
        valid_data = [line.split() for line in f.read().split('\n')[:-1]]
    with open(os.path.join(params.main_dir, 'data', params.dataset, 'test.txt')) as f:
        test_data = [line.split() for line in f.read().split('\n')[:-1]]

    train_tails = set([d[2] for d in train_data])
    train_heads = set([d[0] for d in train_data])
    train_ent = train_tails.union(train_heads)
    train_rels = set([d[1] for d in train_data])

    filtered_valid_data = []
    for d in valid_data:
        if d[0] in train_ent and d[1] in train_rels and d[2] in train_ent:
            filtered_valid_data.append(d)
        else:
            train_data.append(d)
            train_ent = train_ent.union(set([d[0], d[2]]))
            train_rels = train_rels.union(set([d[1]]))

    filtered_test_data = []
    for d in test_data:
        if d[0] in train_ent and d[1] in train_rels and d[2] in train_ent:
            filtered_test_data.append(d)
        else:
            train_data.append(d)
            train_ent = train_ent.union(set([d[0], d[2]]))
            train_rels = train_rels.union(set([d[1]]))

    data_dir = os.path.join(params.main_dir, 'data/{}'.format(params.dataset))
    write_to_file(os.path.join(data_dir, 'train.txt'), train_data)
    write_to_file(os.path.join(data_dir, 'valid.txt'), filtered_valid_data)
    write_to_file(os.path.join(data_dir, 'test.txt'), filtered_test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move new entities from test/valid to train')
    parser.add_argument("--dataset", "-d", type=str, default="fb237_v1",help="Dataset string")
    params = parser.parse_args()
    params.main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    main(params)

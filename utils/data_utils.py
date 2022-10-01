import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import torch
from operator import itemgetter


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())
    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)

def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id
    triplets = {}
    triplets_S_Q = {}
    one2one = {}
    one2many = {}
    many2one = {}
    many2many = {}

    ent = 0
    rel = 0
    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    degree_hr = torch.zeros(ent, len(relation2id), dtype=torch.long)
    degree_tr = torch.zeros(ent, len(relation2id), dtype=torch.long)

    for h, t, r in triplets['train']:
        degree_hr[h, r] += 1
        degree_tr[t, r] += 1

    is_to_one = degree_hr.sum(dim=0).float() / (degree_hr > 0).sum(dim=0) < 1.5
    is_one_to = degree_tr.sum(dim=0).float() / (degree_tr > 0).sum(dim=0) < 1.5

    one2one_index = np.where((is_one_to & is_to_one).numpy())[0]
    one2one['train'] = np.array(index_to_triple(one2one_index, id2relation, triplets['train']))
    one2one['valid'] = np.array(index_to_triple(one2one_index, id2relation, triplets['valid']))

    one2many_index = np.where((is_one_to & ~is_to_one).numpy())[0]
    one2many['train'] = np.array(index_to_triple(one2many_index, id2relation, triplets['train']))
    one2many['valid'] = np.array(index_to_triple(one2many_index, id2relation, triplets['valid']))

    many2one_index = np.where((~is_one_to & is_to_one).numpy())[0]
    many2one['train'] = np.array(index_to_triple(many2one_index, id2relation, triplets['train']))
    many2one['valid'] = np.array(index_to_triple(many2one_index, id2relation, triplets['valid']))

    many2many_index = np.where((~is_one_to & ~is_to_one).numpy())[0]
    many2many['train'] = np.array(index_to_triple(many2many_index, relation2id, triplets['train']))
    many2many['valid'] = np.array(index_to_triple(many2many_index, relation2id, triplets['valid']))

    triplets_S_Q['one'] = one2one
    triplets_S_Q['one2'] = one2many
    triplets_S_Q['many'] = many2one
    triplets_S_Q['many2'] = many2many

    adj_list = []
    triplets_list = np.concatenate((triplets['train'], triplets['valid']))
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets_list[:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets_list[:, 0][idx].squeeze(1), triplets_list[:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    print(len(one2one['train']), len(one2many['train']), len(many2one['train']), len(many2many['train']))
    print(len(one2one['valid']), len(one2many['valid']), len(many2one['valid']), len(many2many['valid']))
    return adj_list, triplets_S_Q, entity2id, relation2id, id2entity, id2relation

def process_files_test(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id
    triplets = {}
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    if 'valid' in files.keys():
        triplets_list = np.concatenate((triplets['train'], triplets['valid']))
    else:
        triplets_list = triplets['train']

    for i in range(len(relation2id)):
        idx = np.argwhere(triplets_list[:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets_list[:, 0][idx].squeeze(1), triplets_list[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation

def index_to_triple(t_index, id2relation, triplets):
    Triple = []
    if len(t_index) == 0:
        Triple = None
    else:
        for i in range(len(triplets)):
            for index in t_index:
                if triplets[i][2] == index:
                    Triple.append([triplets[i][0], triplets[i][1], triplets[i][2]])
    return Triple

def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid_mix(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 100 shards
    # split the first half of the dataset into 100 shards of 300 imgs each
    # use the first half as non-idd and the second half as iid
    combine_weights_train_dataset_count = 1000
    num_shards, num_imgs = 100, 295
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_noiid = np.arange(num_shards*num_imgs)
    labels_noiid = dataset.train_labels.numpy()[:num_shards*num_imgs]
    idxs_full = np.arange(len(dataset))

    # sort labels
    idxs_labels = np.vstack((idxs_noiid, labels_noiid))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_noiid = idxs_labels[0, :]
    idxs_noiid_labels = idxs_labels[1, :]

    idxs_iid = np.arange(30000, len(dataset))

    # divide and assign shards/client
    for i in range(num_users):
        if(i == -1):
            rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_noiid[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                
            print(f"len(dict_users[{i}] 1: ", len(dict_users[i]))
            iid_set = set(np.random.choice(idxs_iid, int(len(dataset) / num_users / 2), replace=False))
            idxs_iid = list(set(idxs_iid) - iid_set) 
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(iid_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))
        elif(i in range(0, 5)):
            rand_set = set(np.random.choice(idx_shard, int(20), replace=False))
            labels = []
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_noiid[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                labels.append(idxs_noiid_labels[rand*num_imgs:(rand+1)*num_imgs])
            idxs_full = list(set(idxs_full) - set(dict_users[i]))
            # print all labels occured in this client
            print("labels: ", np.unique(np.array(labels)))
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        elif(i in range(5, 10)):
            iid_set = set(np.random.choice(idxs_iid, int((len(dataset) - combine_weights_train_dataset_count) / num_users), replace=False))
            idxs_iid = list(set(idxs_iid) - iid_set)
            idxs_full = list(set(idxs_full) - iid_set)
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(iid_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        print("remaining number o idx_iid: ", len(idxs_full))
                

    return dict_users, idxs_full



def mnist_noniid_transformed(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 200 shards
    combine_weights_train_dataset_count = 1000
    num_shards, num_imgs = 100, 295
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_transformed = np.arange(num_shards*num_imgs)
    labels_transformed = dataset.train_labels.numpy()[:num_shards*num_imgs]
    idxs_full = np.arange(len(dataset))

    # sort labels
    idxs_original = np.arange(30000, len(dataset))
    labels_original = dataset.train_labels.numpy()[30000:len(dataset)]

    # print the first 10 labels of labels_transformed and labels_original
    print("labels_transformed: ", labels_transformed[:10])
    print("labels_original: ", labels_original[:10])

    # divide and assign shards/client
    for i in range(num_users):
        if(i in range(0, 5)):
            rand_set = set(np.random.choice(idx_shard, int(20), replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_transformed[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            idxs_full = list(set(idxs_full) - set(dict_users[i]))
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))
        elif(i in range(5, 10)):
            original_set = set(np.random.choice(idxs_original, int((len(dataset) - combine_weights_train_dataset_count) / num_users), replace=False))
            idxs_original = list(set(idxs_original) - original_set)
            idxs_full = list(set(idxs_full) - original_set)
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(original_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        print("remaining number o idx_iid: ", len(idxs_full))

    return dict_users, idxs_full


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  300 imgs/shard X 200 shards

    # save the id of the last 1000 imgs in idxs_full
    idxs_full = np.arange(len(dataset))[-1000:]

    num_shards, num_imgs = 200, 295
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()[:num_shards*num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(200 / num_users), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users, idxs_full

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """

    idxs_full = np.arange(len(dataset))[-1000:]

    num_shards, num_imgs = 200, 245
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)[:num_shards*num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(200 / num_users), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users, idxs_full


def cifar_noniid_transformed(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs -->  250 imgs/shard X 200 shards
    combine_weights_train_dataset_count = 1000
    num_shards, num_imgs = 100, 245
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_transformed = np.arange(num_shards*num_imgs)
    labels_transformed = dataset.train_labels.numpy()[:num_shards*num_imgs]
    idxs_full = np.arange(len(dataset))

    # sort labels
    idxs_original = np.arange(25000, len(dataset))
    labels_original = dataset.train_labels.numpy()[25000:len(dataset)]

    # print the first 10 labels of labels_transformed and labels_original
    print("labels_transformed: ", labels_transformed[:10])
    print("labels_original: ", labels_original[:10])

    # divide and assign shards/client
    for i in range(num_users):
        if(i in range(0, 5)):
            rand_set = set(np.random.choice(idx_shard, int(20), replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_transformed[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            idxs_full = list(set(idxs_full) - set(dict_users[i]))
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))
        elif(i in range(5, 10)):
            original_set = set(np.random.choice(idxs_original, int((len(dataset) - combine_weights_train_dataset_count) / num_users), replace=False))
            idxs_original = list(set(idxs_original) - original_set)
            idxs_full = list(set(idxs_full) - original_set)
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(original_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        print("remaining number o idx_iid: ", len(idxs_full))

    return dict_users, idxs_full



def cifar_noniid_mix(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs -->  250 imgs/shard X 200 shards
    combine_weights_train_dataset_count = 1000
    num_shards, num_imgs = 100, 245
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_noiid = np.arange(num_shards*num_imgs)
    labels_noiid = dataset.train_labels.numpy()[:num_shards*num_imgs]
    idxs_full = np.arange(len(dataset))

    # sort labels
    idxs_labels = np.vstack((idxs_noiid, labels_noiid))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_noiid = idxs_labels[0, :]
    idxs_noiid_labels = idxs_labels[1, :]

    idxs_iid = np.arange(25000, len(dataset))

    # divide and assign shards/client
    for i in range(num_users):
        if(i == -1):
            rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_noiid[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                
            print(f"len(dict_users[{i}] 1: ", len(dict_users[i]))
            iid_set = set(np.random.choice(idxs_iid, int(len(dataset) / num_users / 2), replace=False))
            idxs_iid = list(set(idxs_iid) - iid_set) 
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(iid_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))
        elif(i in range(0, 5)):
            rand_set = set(np.random.choice(idx_shard, int(20), replace=False))
            labels = []
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_noiid[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                labels.append(idxs_noiid_labels[rand*num_imgs:(rand+1)*num_imgs])
            idxs_full = list(set(idxs_full) - set(dict_users[i]))
            # print all labels occured in this client
            print("labels: ", np.unique(np.array(labels)))
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        elif(i in range(5, 10)):
            iid_set = set(np.random.choice(idxs_iid, int((len(dataset) - combine_weights_train_dataset_count) / num_users), replace=False))
            idxs_iid = list(set(idxs_iid) - iid_set)
            idxs_full = list(set(idxs_full) - iid_set)
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(iid_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        print("remaining number o idx_iid: ", len(idxs_full))

    return dict_users, idxs_full
            


def cifar_noniid_mix_original(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs -->  250 imgs/shard X 200 shards
    combine_weights_train_dataset_count = 1000
    num_shards, num_imgs = 100, 245
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_noiid = np.arange(num_shards*num_imgs)
    # labels_noiid = dataset.train_labels.numpy()[:num_shards*num_imgs]
    labels_noiid = np.array(dataset.targets)[:num_shards*num_imgs]
    idxs_full = np.arange(len(dataset))

    # sort labels
    idxs_labels = np.vstack((idxs_noiid, labels_noiid))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_noiid = idxs_labels[0, :]
    idxs_noiid_labels = idxs_labels[1, :]

    idxs_iid = np.arange(25000, len(dataset))

    # divide and assign shards/client
    for i in range(num_users):
        if(i == -1):
            rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_noiid[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                
            print(f"len(dict_users[{i}] 1: ", len(dict_users[i]))
            iid_set = set(np.random.choice(idxs_iid, int(len(dataset) / num_users / 2), replace=False))
            idxs_iid = list(set(idxs_iid) - iid_set) 
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(iid_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))
        elif(i in range(0, 5)):
            rand_set = set(np.random.choice(idx_shard, int(20), replace=False))
            labels = []
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs_noiid[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
                labels.append(idxs_noiid_labels[rand*num_imgs:(rand+1)*num_imgs])
            idxs_full = list(set(idxs_full) - set(dict_users[i]))
            # print all labels occured in this client
            print("labels: ", np.unique(np.array(labels)))
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        elif(i in range(5, 10)):
            iid_set = set(np.random.choice(idxs_iid, int((len(dataset) - combine_weights_train_dataset_count) / num_users), replace=False))
            idxs_iid = list(set(idxs_iid) - iid_set)
            idxs_full = list(set(idxs_full) - iid_set)
            dict_users[i] = np.concatenate(
                    (dict_users[i], np.array(list(iid_set))), axis=0)
            print(f"len(dict_users[{i}]) 2: ", len(dict_users[i]))

        print("remaining number o idx_iid: ", len(idxs_full))

    return dict_users, idxs_full


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

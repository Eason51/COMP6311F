#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, WeightAverage
from utils import get_dataset, average_weights, exp_details, model_register_parametrization, model_remove_parametrizations
import torch.nn.utils.parametrize as parametrize

import random
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np




if __name__ == '__main__':

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'
    
    #! print current device
    print('Current device is ' + device)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    
    
    
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

            
            
            
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # use 10% of test_dataset as combine_weights training set, and the rest are used for testing
    test_dataset, combine_weights_train_dataset = random_split(test_dataset, [int(len(test_dataset) * 0.9), int(len(test_dataset) * 0.1)])

    if(args.weighted):
        # combine_weights = torch.nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]), requires_grad=True)
        combine_weights = copy.deepcopy(global_model)
        combine_weights.alpha = torch.nn.Parameter(torch.ones(int(args.num_users * args.frac)) / int(args.num_users * args.frac))
        combine_weights.to(device)

        combine_weights_optimizer = torch.optim.SGD([combine_weights.alpha], lr=0.01)
        # combine_weights_optimizer = torch.optim.Adam([combine_weights.alpha], lr=0.01)
        combine_weights_criterion = torch.nn.NLLLoss()
        combine_weights_record = []
        combine_weights_record.append(copy.deepcopy(combine_weights.alpha))

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    global_control = {}
    for key in global_weights.keys():
        global_control[key] = torch.zeros(global_weights[key].shape).to(device)

    local_controls = []
    for i in range(args.num_users):
        local_controls.append(copy.deepcopy(global_control))

    delta_weights = []
    for i in range(args.num_users):
        delta_weights.append(copy.deepcopy(global_weights))

    delta_controls = []
    for i in range(args.num_users):
        delta_controls.append(copy.deepcopy(global_control))


    # Training
    train_loss, train_accuracy = [], []
    test_loss_list, test_accuracy_list = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, 
                global_control=global_control, local_control=local_controls[idx],
                delta_weight=delta_weights[idx], delta_control=delta_controls[idx])
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        aggregated_delta_control = {}
        for key in global_control.keys():
            aggregated_delta_control[key] = torch.zeros(global_control[key].shape).to(device)
        
        for idx in idxs_users:
            for key in delta_controls[idx].keys():
                aggregated_delta_control[key] += delta_controls[idx][key] / len(idxs_users)

        for key in global_control.keys():
            global_control[key] += aggregated_delta_control[key] * (len(idxs_users) / args.num_users)

        # update global weights
        if(not args.weighted):
            global_weights = average_weights(local_weights)

        # train combined weights by averaging local weights through the weighted average
        else:
            combine_weights_batch_size = 1000
            # make a dataloader for the combine_weights_train_dataset
            combine_weights_train_dataloader = DataLoader(combine_weights_train_dataset, batch_size=combine_weights_batch_size, shuffle=True)
            for batch_idx, (images, labels) in enumerate(combine_weights_train_dataloader):
                images, labels = images.to(device), labels.to(device)
                combine_weights_optimizer.zero_grad()

                model_register_parametrization(combine_weights, local_weights, combine_weights.alpha)

                output = combine_weights(images)
                loss = combine_weights_criterion(output, labels)
                loss.backward()
                combine_weights_optimizer.step()

                # ensure alpha is always positive
                combine_weights.alpha.data = torch.abs(combine_weights.alpha.data)
                # normalize alpha so that it is always a probability distribution
                combine_weights.alpha.data = torch.div(combine_weights.alpha.data, torch.sum(combine_weights.alpha.data))

            combine_weights_record.append(copy.deepcopy(combine_weights.alpha))



        # update global weights 
        if(not args.weighted):
            global_model.load_state_dict(global_weights)
        else:
            model_remove_parametrizations(combine_weights)
            global_model.load_state_dict(combine_weights.state_dict(), strict=False)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)


    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    if(args.weighted):
        print("combine_weights_record: ", combine_weights_record)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_Scaf{}_Wt{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.scaffold, args.weighted, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_Scaf{}_Wt{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.scaffold, args.weighted, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    
    # Plot test Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, color='k')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_Scaf{}_Wt{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_acc.png'.
                format(args.scaffold, args.weighted, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    
    # # Plot test Loss vs Communication rounds
    # plt.figure()
    # plt.title('Test Loss vs Communication rounds')
    # plt.plot(range(len(test_loss_list)), test_loss_list, color='k')
    # plt.ylabel('Test Loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_Scaf{}_Wt{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_loss.png'.
    #             format(args.scaffold, args.weighted, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

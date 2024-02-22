'''
Implementation of evaluation model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import torch
from src.methods.shared.named_loss import get_named_loss
from src.methods.shared.named_methods import get_named_method
from src.methods.shared.named_metric import get_named_metric
from torch.utils.data import DataLoader

from src.data.get_dataset import get_dataset


def compute_loss(directory, args ):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # set fixed seed
    torch.manual_seed(args["random_seed"])
    # np.random.seed(0)  # init random seed

    random_state = np.random.RandomState(args["random_seed"])
    data_seed = random_state.randint(2 ** 31)

    # get loss function
    criterion = get_named_loss(args["loss"])

    # load entire dataset since the task is to learn a representation
    train_dl, args = get_dataset(args)

    if args["batch_size"] > 8:
        args["batch_size"] = 8

    if args["multithread"]:

        train_dl = DataLoader(train_dl, batch_size=args["batch_size"], shuffle=False, num_workers=16, drop_last=False, pin_memory=False)

        print("Using Dataloader multithreading!")
    else:
        train_dl = DataLoader(train_dl, batch_size=args["batch_size"], shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
        print("Not using Dataloader multithreading!")

    args["repetitions"] = 1024

    # get model
    model = get_named_method(args["method"])(**args, criterion=criterion)
    model.load_state(os.path.join(directory, "checkpoint", "model.pth"))

    # move model to gpu
    model.to(device)

    print("===============START EVALUATION ELBO===============")

    elbo_list = []
    reconstruction_list = []

    i = 0
    for images, classes in train_dl:

        if i >= args["repetitions"]:
            break


        model.eval()

        # move to GPU
        images = images.to(device)

        # compute the model output calculate loss
        loss, _ = model.compute_loss(images)

        elbo_list.append(loss["elbo"].item())
        reconstruction_list.append((loss["reconstruction"].item()))

    scores = {}
    scores["elbo"] = np.mean(elbo_list)
    scores["reconstruction"] =  np.mean(reconstruction_list)
    return scores




def create_evaluation_directory(directory):

    process_dir = os.path.join(directory, "evaluation")

    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)
    #else:
        #raise FileExistsError("Evaluation folder exists")

    return process_dir


def evaluation_model(directory, args):

    representation_directory = os.path.join(directory, "postprocess")
    model_directory = os.path.join(directory, "model")

    # create the folder devoted to the postprocessing
    directory = create_evaluation_directory(directory)

    print("===============START EVALUATION===============")

    for name in args["metrics"]:

        # create metric dir
        metric_dir = os.path.join(directory, name)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        else:
            continue

        metric = get_named_metric(name)

        for mode in args["mode_eval"]:

            # create metric mode dir
            metric_mode_dir = os.path.join(metric_dir, mode)
            if not os.path.exists(metric_mode_dir):
                os.makedirs(metric_mode_dir)

            representation_path = os.path.join(representation_directory, "representations")
            classes_path = os.path.join(representation_directory, "classes")

            for noise in args["label_noise"]:
                noise_path = os.path.join(classes_path, noise)
                labels = os.path.join(noise_path, "classes")


                metric_mode = metric( mode = mode, representation_path= representation_path,  classes_path= labels )

                dict_score = metric_mode.get_score() # score wrt hyperparameters

                # create metric mode dir
                metric_mode_noise_dir = os.path.join(metric_mode_dir, noise)
                if not os.path.exists(metric_mode_noise_dir):
                    os.makedirs(metric_mode_noise_dir)

                # save scores as dictionary with the hyperparameters as keys
                with open(os.path.join(metric_mode_noise_dir, 'scores.pkl'), 'wb') as handle:
                    pickle.dump(dict_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # log
                print("Metric: {}, Mode: {}, Noise: {}, Scores: {}".format(name, mode, noise, dict_score))



    loss_path = os.path.join(directory, "loss")
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

        loss_scores = compute_loss(model_directory, args)


        print("Elbo: {}".format(loss_scores["elbo"]))
        print("Reconstruction: {}".format(loss_scores["reconstruction"]))

        if not os.path.exists(loss_path):
            os.makedirs(loss_path)

        # save scores as dictionary with the hyperparameters as keys
        with open(os.path.join(loss_path, 'loss_scores.pkl'), 'wb') as handle:
            pickle.dump(loss_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("===============END EVALUATION===============")

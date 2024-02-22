'''
Implementation of general training scheme
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device_type = "cuda" if use_cuda else "cpu"
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import pickle
import numpy as np
import time
import pandas as pd

from src.methods.shared.named_methods import get_named_method
from src.methods.shared.named_loss import get_named_loss
from src.methods.shared.postprocessing.postprocess import get_representation_dataloader
from src.data.get_dataset import get_dataset


def save_model(directory, model):
    checkpoint_dir = os.path.join(directory, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model.pth')  # _checkpoint_{iteration + 1}
    model.save_state( checkpoint_path)
    return checkpoint_path


def save_representation_checkpoint(args, train_dl, dataset, model, iteration, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model.eval()
    representation_to_save, classes_to_save = get_representation_dataloader(args, train_dl, model, args["factor_idx"])

    # save representation
    np.savez_compressed(os.path.join(directory, "representations_{}".format(iteration)), **representation_to_save)
    pd.DataFrame(classes_to_save, columns=[dataset.factor_names[i] for i in args["factor_idx"]]).to_csv(
        os.path.join(directory, "classes_{}.csv".format(iteration)), index=False)





def train_model(directory, args):

    # set fixed seed
    torch.manual_seed(args["random_seed"])

    # Create a numpy random state. We will sample the random seeds for training
    random_state = np.random.RandomState(args["random_seed"])
    data_seed = random_state.randint(2 ** 31)

    args["data_seed"] = data_seed

    # the folder devoted to the model already exists
    #directory = create_model_directory(directory)

    criterion = get_named_loss(args["loss"])
    optimizer = torch.optim.Adam

    # load entire dataset since the task is to learn a representation
    train_dataset, args = get_dataset(args)



    # get model
    directory = os.path.join(directory, "model")
    model = get_named_method(args["method"])(**args, criterion=criterion)


    # build optimizer with model parameters
    model.build_model(optimizer, **args)

    # move model to gpu
    model.to(device)

    model.load_state(os.path.join(directory, "checkpoint", "model.pth"))

    print("Number of total parameters of the model: {:,}".format(model.num_params()))
    print("Number of trainable parameters of the model: {:,}".format(model.num_trainable_params()))
    print("Usinge device: ", device, "| Device type: ", device_type)


    # list to save statistics
    loss_list = []

    # min loss
    best_loss = np.inf

    save_interval = 10000  # Save model every 10,000 iterations
    n_accumulation = args["grad_acc_steps"]  # steps for gradient accumulation
    total_iterations = args["iterations"] * n_accumulation  # count n_accumulations as one iteration

    if args["multithread"]:

        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"]//n_accumulation, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

        print("Using Dataloader multithreading!")
    else:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"]//n_accumulation, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
        print("Not using Dataloader multithreading!")

    # Create an iterator for the DataLoader
    data_iter = iter(train_dl)

    print("===============START TRAINING===============")



    acc_loss = {}  # accumulated loss
    batch_loss = 0.0
    batch_iterations = 1
    iterations_loss = 0.0
    iterations_iterations = 1
    start_time = time.time()

    for iteration in range(total_iterations):

        try:
            inputs, _ = next(data_iter)
        except StopIteration:
            # If we have reached the end of the DataLoader, create a new iterator.
            data_iter = iter(train_dl)
            inputs, _ = next(data_iter)
            print("-" * 20, "New epoch!", "-" * 20)

            # update learning rate
            # model.update_learning_rate(batch_loss/batch_iterations)

            batch_loss = 0.0  # intialize
            batch_iterations = 1

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()

        # move data to GPU
        inputs = inputs.to(device)


        # compute the model output calculate loss
        loss, y_hat = model.compute_loss(inputs)

        # Backpropagation and optimization
        model.backward(loss["loss"])

        # if dict is empty
        if not (acc_loss):
            acc_loss = {name: value.item() for name, value in loss.items()}
        else:
            # sum
            for name, value in loss.items():
                acc_loss[name] += value.item()


        if (iteration + 1) % n_accumulation == 0:
            # update weights with clipped gradients
            model.update_weights()

            # clear the gradients
            model.zero_grad()

            loss_list.append(
                {name: value / n_accumulation for name, value in acc_loss.items()})  # add loss for each iteration
            acc_loss = {}  # initilize accumulator

            batch_loss += loss_list[-1]["loss"]
            batch_iterations += 1
            iterations_iterations += 1
            iterations_loss += loss_list[-1]["loss"]


        # save if improved loss
        if (iteration + 1) % save_interval == 0:


            iterations_loss = iterations_loss / save_interval
            if (iterations_loss) < best_loss:
                print(f"Loss improvement! New best loss: {iterations_loss:.2f}, Old best loss: {best_loss:.2f}")
                best_loss = iterations_loss
                iterations_iterations = 1
                iterations_loss = 0.0

                checkpoint_path = save_model(directory, model)
                train_loss = loss_list[-1]["loss"]
                train_rec_loss = loss_list[-1]["reconstruction"]
                train_kl_loss = loss_list[-1]["kl"]
                print(
                    f"Iteration [{iteration + 1}/{total_iterations}], ELBO: {train_loss:.2f}, Reconstruction: {train_rec_loss:.2f}, KL: {train_kl_loss:.2f}, Checkpoint saved at {checkpoint_path}")

                print("Save representation!")
                # save representation
                save_representation_checkpoint(args, train_dl, train_dataset, model, iteration,
                                               os.path.join(directory, "intermediate_representations"))

            else:
                print("Loss not improved")

            # Calculate and print the time taken for this checkpoint
            elapsed_time = time.time() - start_time
            print(f"Time elapsed for last {save_interval} iterations: {elapsed_time:.2f} seconds")
            remainig_time = (total_iterations - (iteration + 1)) / save_interval * elapsed_time
            print(f"Time to complete training {total_iterations} iterations: {remainig_time:.2f} seconds")

            start_time = time.time()

    print("===============END TRAINING===============")

    for k in loss_list[0].keys():

        # save loss history
        with open(os.path.join(directory, f'{k}.txt'), 'w') as f:
            for line in loss_list:
                f.write(f"{line[k]}\n")

        # saving info file

    model_path, _ = os.path.split(directory)
    if "model_num" not in args:
        model_path, _ = os.path.split(directory)
        _, model_num = os.path.split(model_path)
    else:
        model_num = args["model_num"]

    args["elbo"] = loss_list[-1]["elbo"]
    args["reconstruction"] = loss_list[-1]["reconstruction"]
    args["model_num"] = int(model_num)

    with open(os.path.join(model_path, 'info.pkl'), 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # free gpu memory

    del model
    del train_dataset
    del train_dl
    torch.cuda.empty_cache()


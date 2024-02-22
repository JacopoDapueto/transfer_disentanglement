from __future__ import division, absolute_import, print_function

from src.data.named_data import get_named_data


def get_dataset(args):

    # load entire dataset since the task is to learn a representation
    train_dataset = get_named_data(args["dataset"])(latent_factor_indices=args["factor_idx"], batch_size=1,
                                                    random_state=args["data_seed"],
                                                    resize=args["resize"],
                                                    center_crop=args["center_crop"])

    # prepare arguments for next functions
    args["n_channel"] = train_dataset.num_channels()
    args["data_shape"] = train_dataset.get_shape()
    return train_dataset, args

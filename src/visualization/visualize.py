'''
Implementation of visualizations
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

from src.methods.shared.named_methods import get_named_method
from src.methods.shared.named_loss import get_named_loss

from src.methods.shared.utils.utils import batch

from src.methods.shared.utils.visualize_utils import *
from src.data.get_dataset import get_dataset


def representation_rank(template, representations, k=10):
    """Return k most similar representations idexes similar to template"""

    distances = np.absolute(representations - template)
    sorted_idx = np.argsort(distances)[:k] # descending order, first k
    return sorted_idx


def rank_each_dimensions(template, to_rank, k=11):

    rank = []

    for j in range(template.shape[0]):
        j_rank = representation_rank(template[j], to_rank[:, j], k=k)
        rank.append(j_rank)

    return  rank


def rank_entire_dataset( args, templates, train_dl, model, k=11):
    # iterate over the entire dataset and return k-nerest image per template (for each mode)
    top_k_representation = {mode: { i: None for i, _ in enumerate(templates)} for mode in args["mode"]}

    top_k = {mode: { i: None for i, _ in enumerate(templates)} for mode in args["mode"]} # the top k for each mode and template


    # compute representation
    template_representations = model.encode(templates)

    # iterate over the dataset, with sampling
    for idx in batch(range(train_dl.num_images()), args["batch_size"]):

        images, _ = train_dl.get_images(idx)

        # move data to GPU
        images = images.to(device)

        dict_representation = model.encode(images)

        # update representation list
        for mode in args["mode"]:
            representations = dict_representation[mode].cpu().detach().numpy()
            template_representations_mode = template_representations[mode].cpu().detach().numpy()
            print(template_representations_mode.shape)
            top_k_mode = top_k[mode]
            for i, template in enumerate(template_representations_mode):
                top_k_mode_template = top_k_mode[i]

                # create empty rank
                if top_k_mode_template is not None:
                    top_k_mode_template = train_dl.get_images(idx)
                    #top_k[mode][i] = [[] for j in range(template.shape[0]) ]

                representations_to_rank = representations if top_k_mode_template is None else np.vstack((top_k_mode_template, representations))

                rank_dim = rank_each_dimensions(template, representations_to_rank, k=k)

                #top_k_representation
                top_k[mode][i] = [ idx[dim] for dim in rank_dim]

    return top_k

def save_k_similarity_mode(directory, mode, templates, rank, train_dl):
    results_dir = os.path.join(directory, "template_similarity", mode)

    if not os.path.exists(results_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(results_dir)

    # sample random images
    rank_mode = rank[mode]

    for i, template in enumerate(templates):

        template_representation = representations[i]

        for j in range(template_representation.shape[0]):
            # exclude template
            j_representations = np.vstack((representations[:i], representations[i + 1:]))
            j_rank = representation_rank(template_representation[j], j_representations[:, j], k=k)
            rank.append(images[j_rank, ...])

        rank = np.moveaxis(np.array(rank), 2, -1)
        template = np.moveaxis(template, 0, -1)

        save_rank(template, rank, os.path.join(results_dir, "rank_{}.jpg".format(i)))


def save_k_similarity(args, directory, train_dl, model, n_templates=10, k=11):
    # sample random images
    idx = np.random.choice(range(train_dl.num_images()), size=n_templates)
    templates, _ = train_dl.get_images(idx)

    rank_modes = rank_entire_dataset(args, templates, train_dl, model, k)
    for mode in args["mode"]:
        rank = rank_modes[mode]
        save_k_similarity_mode(directory, mode, templates, rank, train_dl)


def save_random_samples(directory, random_samples):
    '''

    :param directory:
    :param random_samples:
    :return:
    '''

    #random_samples = np.expand_dims(random_samples.squeeze(axis=1), axis=3)
    random_samples = np.moveaxis(random_samples, 1, -1)

    results_dir = os.path.join(directory, "random_samples")

    if not os.path.exists(results_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(results_dir)

    grid_save_images(random_samples, os.path.join(results_dir, "random_samples.jpg"))


def save_reconstruction(directory, images, reconstruction):
    '''

    :param directory:
    :param images:
    :param reconstruction:
    :return:
    '''


    images = np.moveaxis(images, 1, -1)
    reconstruction = np.moveaxis(reconstruction, 1, -1)

    paired_pics = np.concatenate((images, reconstruction), axis=2)
    paired_pics = [paired_pics[i, :, :, :] for i in range(paired_pics.shape[0])]

    results_dir = os.path.join(directory, "reconstructions")

    if not os.path.exists(results_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(results_dir)

    # save visualizations as images
    grid_save_images(paired_pics, os.path.join(results_dir, "reconstruction.jpg"))


def save_pairwise_representations():
    pass



def save_loss_plot(directory):
    '''

    :param directory:
    :return:
    '''


    files = []
    names = []
    for file in os.listdir(os.path.join(directory, "model")):
        if file.endswith(".txt"):
            files.append(file)
            names.append(os.path.splitext(file)[0])

    for file, name in zip(files, names):
        #  loading loss plot
        f = open(os.path.join(directory, "model", file), "r")

        loss_list = []
        for x in f:
            loss_list.append(np.float64(x))
        f.close()

        plt.plot(range(len(loss_list)), loss_list)
        plt.ylabel(name)
        plt.xlabel("Step")

        # saving loss plot
        results_dir = os.path.join(directory, "visualizations","loss", name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(os.path.join(results_dir, f"{name}.jpg"))
        plt.clf()  # clear figure

        # save cut plot after 5000 steps
        start_step = 5000
        plt.plot(range(start_step, len(loss_list)), loss_list[start_step:])
        plt.ylabel(name)
        plt.xlabel("Step")

        # saving loss plot
        results_dir = os.path.join(directory, "visualizations","loss", name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(os.path.join(results_dir, f"cut_{name}.jpg"))
        plt.clf()  # clear figure


def create_visualization_directory(directory):
    '''

    :param directory:
    :return:
    '''

    visualization_dir = os.path.join(directory, "visualizations")

    # make experiment directory
    if not os.path.exists(visualization_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(visualization_dir)
    else:
        raise FileExistsError("Visualization folder exists")

    return visualization_dir


def visualize_model(directory, args):
    '''

    :param directory:
    :param args:
    :return:
    '''

    # set fixed seed
    torch.manual_seed(args["random_seed"])
    np.random.seed(0)  # init random seed

    # create the folder devoted to the postprocessing
    old_directory = directory
    directory = create_visualization_directory(directory)

    # plot the loss over steps
    save_loss_plot(old_directory)

    # get loss function
    criterion = get_named_loss(args["loss"])

    train_dataset, args = get_dataset(args)

    # get model
    model = get_named_method(args["method"])(**args, criterion=criterion)


    # load pretrained weights
    model.load_state(os.path.join(old_directory, "model", "checkpoint", "model.pth"))


    # move model to gpu
    model.to(device)

    print("===============START VISUALIZATIONS===============")

    num_random_samples = 16
    num_animations = 5
    num_templates = 10
    args["batch_size"] = 16

    with torch.no_grad():
        # the dataset requires a dataloader
        if args["multithread"]:
            train_dl = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

        else:
            # save one batch
            train_dl = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0, drop_last=False, pin_memory=False)

        data_iter = iter(train_dl)
        images_reconstruction, _ = next(data_iter)
        images_traversal, _ = next(data_iter)


        # move data to GPU
        images_reconstruction = images_reconstruction.to(device)

        # compute the model output calculate loss
        # save recontruction of images
        loss, reconstruction = model.compute_loss(images_reconstruction)
        save_reconstruction(directory, images_reconstruction.cpu().detach().numpy(), reconstruction.cpu().detach().numpy())

        # save random samples
        random_samples = model.sample(num_random_samples)
        save_random_samples(directory, random_samples.cpu().detach().numpy())

        # save latent traversals
        images_traversal = images_traversal.to(device)
        dict_representation = {k: r.cpu().detach().numpy() for k, r in model.encode(images_traversal).items()}

        #  save animation traversal
        for mode in args["mode"]:
            save_traversal(directory, mode, dict_representation[mode], model)

    # image similarity wrt a dimension
    #save_k_similarity(args, directory, train_dl, model, num_templates)


    print("===============END VISUALIZATIONS===============")
    # free gpu memory
    del model
    torch.cuda.empty_cache()

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import random

def plot_cov_factors_from_matrix( covs, path):

    sns.set(font_scale=2)  # font size 2
    plt.figure(figsize=(4, 4))
    covs = np.round(covs, 2)
    ax = sns.heatmap(covs, yticklabels=[], xticklabels=[], annot=False, cmap="Blues", cbar=False, vmin=0, vmax=1)

    # Drawing the frame
    ax.axhline(y=0, color='red', linewidth=3)
    ax.axhline(y=covs.shape[0], color='red',
                linewidth=3)

    ax.axvline(x=0, color='red',
                linewidth=3)

    ax.axvline(x=covs.shape[1],
                color='red', linewidth=3)

    fig = ax.get_figure()
    fig.tight_layout()
    #plt.xlabel("")
    #plt.ylabel("")
    #plt.title('Relation matrix')
    plt.savefig(os.path.join(path),  dpi=600 ,bbox_inches='tight', pad_inches=0)
    #plt.savefig(os.path.join(path, "./cov_plot.eps"))
    plt.clf()






def clamp(n, min_v, max_v):
    if n < min_v:
        return min_v
    elif n > max_v:
        return max_v
    else:
        return n



def add_correlation(cov, p=0.0, I=0.05):
    modified_matrix = np.copy(cov)  # Create a copy of the original matrix
    rows, cols = cov.shape

    for i in range(rows):
        for j in range(cols):


            if np.random.rand() < p:  # Check if the randomly generated probability is less than p
                # Modify the cell with some value or operation

                #perturbation = -I if np.random.rand() < 0.5 else I
                perturbation = -I if modified_matrix[i, j] > 0.5 else I
                modified_matrix[i, j] += perturbation
                modified_matrix[i, j] = clamp(modified_matrix[i, j], 0.05, 1.0)

    return modified_matrix


def remove_correlation(cov, p=0.0, I=0.01):
    modified_matrix = np.copy(cov)  # Create a copy of the original matrix
    rows, cols = cov.shape

    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < p:  # Check if the randomly generated probability is less than p
                # Modify the cell with some value or operation
                #perturbation = -I if np.random.rand() < 0.5 else I
                perturbation = -I
                modified_matrix[i, j] += perturbation
                modified_matrix[i, j] = clamp(modified_matrix[i, j], 0.01, 1.0)

    return modified_matrix








if __name__ == "__main__":

    N = 8
    F = 5
    ideal = np.eye(N, F)

    p = 0.0035
    I = 0.018

    add_steps = 5000
    reduce_steps = 7000


    # plot almost ideal matrix
    perturbed_cov = np.copy(ideal)
    for s in range(100):
        perturbed_cov = add_correlation(perturbed_cov, p=p, I=I)

    plot_cov_factors_from_matrix(perturbed_cov, os.path.join("output", "almost_ideal.png"))

    # plot medium matrix
    for s in range(add_steps - 100):
        perturbed_cov = add_correlation(perturbed_cov, p=p, I=I)

    plot_cov_factors_from_matrix(perturbed_cov, os.path.join("output", "medium.png"))


    # plot underfitted matrix
    #for s in range(reduce_steps):
        #perturbed_cov = remove_correlation(perturbed_cov, p=p, I=I)

    perturbed_cov = np.zeros((N, F))

    for i in range(N):
        for j in range(F):
            perturbed_cov[i, j] = random.uniform(0.05, 0.3)
    plot_cov_factors_from_matrix(perturbed_cov, os.path.join("output", "underfitted.png"))


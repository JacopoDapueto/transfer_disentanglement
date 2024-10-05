import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random



from src.evaluation.metrics.omes import disentanglement_score
"""
def generate_correlated_vectors(N, M, P, I):
    vectors = []
    for _ in range(N):
        vector = np.random.normal(size=M)  # generate M random numbers
        for i in range(M):
            for j in range(i+1, M):
                if np.random.rand() < P:
                    # if the condition is met, set the correlation between the i-th and j-th elements to I
                    vector[i] = vector[i] * np.sqrt(1 - I**2) + vector[j] * I
                    vector[j] = vector[j] * np.sqrt(1 - I**2) - vector[i] * I
        vectors.append(vector)
    return vectors


# also make  correlate with the j-th vector at i-th element (for correlation of multiple FoV)
def generate_correlated_vectors(N, M, P, I):
    vectors = []
    for _ in range(N):
        vector = np.zeros(M)
        for i in range(M):
            for j in range(i+1, M):
                if np.random.rand() < P:
                    # Generate two correlated random variables
                    x, y = stats.truncnorm.rvs(a=-2, b=2, loc=0, scale=1, size=2)
                    # Apply correlation intensity
                    corr_matrix = np.array([[1, I], [I, 1]])
                    inv_sqrt = np.linalg.inv(np.linalg.cholesky(corr_matrix))
                    x, y = np.dot(inv_sqrt, [x, y])
                    # Assign to vector
                    vector[i] = x
                    vector[j] = y
                else:
                    # Assign uncorrelated random variables
                    vector[i] = np.random.randn()
                    vector[j] = np.random.randn()
        vectors.append(vector)
    return vectors
"""


def plot_cov(cov, factors, path):

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cov, xticklabels=factors, annot=True, cmap="Blues", cbar=False,  vmin=0, vmax=1)

    plt.xlabel("Factors")
    plt.ylabel("Dimensions")
    plt.title('Relation matrix')
    plt.savefig(path)
    plt.clf()


def plot_scores(scores, path):


    plt.figure(figsize=(8, 6), dpi=300)

    plt.ylim(0.001, 1.001)

    plt.scatter(range(len(scores)), scores)
    plt.xlabel("Perturbation intensity")
    plt.ylabel("Disentanglement score")
    plt.title('Scores wrt perturbation')
    plt.savefig(path)
    plt.clf()


def plot_scores_dataframe(scores, path, x="iter", y="modularity"):

    scores = scores.iloc[::-1].reset_index(drop=True)

    # select 3 important points: top, middle, bottom
    top = scores[(scores[y] >=0.80 ) ]#.sort_index().set_index("index")
    middle = scores[(scores[y] >=0.45 ) & (scores[y] <= 0.55)]#.sort_index().set_index("index")
    bottom = scores[(scores[y] <=0.15 ) & (scores[y] >= 0.10)]#.sort_index().set_index("index")

    print(len(top))
    print(len(middle))
    print(len(bottom))

    top_idx = random.choice(top.index)
    middle_idx = random.choice(middle.index)
    bottom_idx = random.choice(bottom.index)

    important = scores.loc[
        [top_idx, middle_idx, bottom_idx]]  # .reset_index().sample(n=3).sort_index().set_index("index")

    # plt.figure(figsize=(8, 6), dpi=600)
    sns.set_style("whitegrid")
    plt.tight_layout()
    sns.set(font_scale=1.75)  # font size 2
    with sns.axes_style("whitegrid"):
        ax =sns.scatterplot(data=scores,x=x, y=y, alpha=0.75, linewidth=0, color="purple", s=80)
        ax.axes.get_xaxis().set_ticks([])


        ax = sns.scatterplot(data=important, x=x, y=y, alpha=1.0, linewidth=1.5, edgecolor="red", color="purple", s=80)
        ax.axes.get_xaxis().set_ticks([])

    #ax.tick_params(left=True, bottom=False)

    ax.invert_xaxis()
    plt.xlabel("Disentanglement intensity", fontsize='large')
    plt.ylabel("OMES", fontsize='large')
    plt.tight_layout()

    plt.ylim(0.001, 1.001)

    plt.savefig(path, dpi=600,bbox_inches='tight')
    plt.clf()


def clamp(n, min_v, max_v):
    if n < min_v:
        return min_v
    elif n > max_v:
        return max_v
    else:
        return n



def add_correlation(cov, p=0.0, I=0.01):
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

"""
def remove_correlation(cov, p=0.0, I=0.02):
    modified_matrix = np.copy(cov)  # Create a copy of the original matrix
    rows, cols = cov.shape

    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < p:  # Check if the randomly generated probability is less than p
                # Modify the cell with some value or operation
                #perturbation = -I if np.random.rand() < 0.5 else I
                perturbation = -I
                modified_matrix[i, j] += perturbation
                modified_matrix[i, j] = clamp(modified_matrix[i, j], 0.05, 1.0)

    return modified_matrix
"""

def remove_correlation(cov, p=0.0, I=0.55):
    modified_matrix = np.copy(cov)  # Create a copy of the original matrix
    rows, cols = cov.shape

    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < p:  # Check if the randomly generated probability is less than p
                # Modify the cell with some value or operation
                #perturbation = -I if np.random.rand() < 0.5 else I
                perturbation = - random.uniform(0.05, I)
                modified_matrix[i, j] += perturbation
                modified_matrix[i, j] = clamp(modified_matrix[i, j], 0.05, 1.0)

    return modified_matrix


def test_modularity(cov, steps=20, reduce_steps=20, p=0.0, I=0.01, factors = None):

    # p is probability of perturbing a dimension with a FoV

    # for each step virtually add noise to representation

    scores = []
    perturbed_cov = np.copy(cov)
    score, _ = disentanglement_score(perturbed_cov, factors, alpha=1.0, gamma=True)
    scores.append(score)

    # add correlation
    for s in range(steps):

        perturbed_cov = add_correlation(perturbed_cov, p=p, I=I)

        score, _ = disentanglement_score(perturbed_cov, factors, alpha=1.0, gamma=True)

        scores.append(score)

    # remove encoding of FoV
    I_list = np.linspace(0.02, 0.10,  reduce_steps, endpoint=True)
    for s in range(reduce_steps):
        perturbed_cov = remove_correlation(perturbed_cov, p=p, I=I_list[s])

        score, _ = disentanglement_score(perturbed_cov, factors, alpha=1.0, gamma=True)

        scores.append(score)


    return perturbed_cov, scores


def test_compactness(cov, steps=20, reduce_steps=20, p=0.0 , I=0.01, factors = None):
    # p is probability of perturbing a dimension with a FoV

    # for each step virtually add noise to representation

    scores = []
    perturbed_cov = np.copy(cov)
    score, _ = disentanglement_score(perturbed_cov, factors, alpha=0.0, gamma=True)
    scores.append(score)

    for s in range(steps):
        perturbed_cov = add_correlation(perturbed_cov, p=p, I=I)

        score, _ = disentanglement_score(perturbed_cov, factors, alpha=0.0, gamma=True)

        scores.append(score)

    I_list = np.linspace(0.05, 0.55, reduce_steps, endpoint=True)
    # remove encoding of FoV
    for s in range(reduce_steps):
        perturbed_cov = remove_correlation(perturbed_cov, p=p,  I=I_list[s])

        score, _ = disentanglement_score(perturbed_cov, factors, alpha=0.0, gamma=True)

        scores.append(score)

    return perturbed_cov, scores



def test_metric(N=10, F=7, p=0.05, add_steps=20, reduce_steps=20, I=0.01, factors=None):


    cov = np.eye(N, F, k=0)

    perturbed_cov, modularity_scores = test_modularity(cov, add_steps, reduce_steps, I=I, p=p, factors=factors)
    plot_scores(modularity_scores, f"./output/score_modularity_{p}_{I}.png")
    plot_cov(perturbed_cov, factors, f"./output/perturbed_modularity_{p}_{I}.png")

    perturbed_cov, compactness_scores = test_compactness(cov, add_steps, reduce_steps, I=I, p=p, factors=factors)
    plot_cov(perturbed_cov, factors, f"./output/perturbed_compactness_{p}_{I}.png")
    plot_scores(compactness_scores, f"./output/score_compactness_{p}_{I}.png")

    return modularity_scores, compactness_scores











if __name__ == "__main__":
    # Example usage:
    #N = 5
    #M = 10_000_000
    #P = 0.5
    #I = 0.8
    #vectors = generate_correlated_vectors(N, M, P, I)

    N =10
    F=7
    p = 0.005
    add_steps= 5000
    reduce_steps = 5000



    #p_list = [0.001]
    #I_list = [0.005]

    #p_list = [0.001, 0.005, 0.01, 0.015, 0.02]
    #I_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

    #p_list = [0.002, 0.003,  0.001, 0.0005]
    #I_list = [  0.015, 0.01, 0.0002, 0.008, 0.005 ] # 0.001, 0.005,

    #p_list = [0.003, 0.004, 0.002, 0.0035, 0.0027]
    #I_list = [  0.015, 0.01,  0.001, 0.005, 0.018, 0.008  ] #

    p_list = [0.003, 0.002, 0.0035]
    I_list = [0.015, 0.01,  0.001, 0.004, 0.005, 0.018, 0.008]  #

    k = 500
    factors = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] #

    scores_dict = {"modularity": [], "compactness":[], "p":[], "I":[], "iter":[]}

    for N in [10]: # , 5, 15
        for F in [7]: # , 5, 10
            for p in p_list:
                for I in I_list:

                    #if  p==0.004 and I ==0.018:
                        #continue

                    #if  p==0.002 and I ==0.0015:
                        #continue

                    modularity_scores, compactness_scores = test_metric(N, F, p, add_steps, reduce_steps, I, factors[:F])

                    scores_dict["modularity"].extend(modularity_scores)
                    scores_dict["compactness"].extend(compactness_scores)

                    scores_dict["p"].extend([ p for _ in range(len(compactness_scores))])
                    scores_dict["I"].extend([ I for _ in range(len(compactness_scores))])
                    scores_dict["iter"].extend([i for i in range(len(compactness_scores))])


    df = pd.DataFrame.from_dict(scores_dict)

    df_sampled = df.reset_index().sample(n=k).sort_index().set_index("index")

    # plot scores
    plot_scores_dataframe(df_sampled, f"./output/scores_modularity.png", y="modularity" )
    plot_scores_dataframe(df_sampled, f"./output/scores_compactness.png", y="compactness")



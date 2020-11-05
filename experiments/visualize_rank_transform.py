import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import random

def score_transform(score_list, score_orig_list, score_transform_type):
    scores = np.asarray(score_list)
    scores_orig = np.asarray(score_orig_list)

    if score_transform_type == 0:
        # convert [1, 0, 5] to [0.2, 0, 1]
        scores = (scores - min(scores)) / (max(scores) - min(scores) + 1e-9)

    elif score_transform_type == 1:
        # convert [1, 0, 5] to [0.5, 0, 1]
        s = np.argsort(scores)
        n = len(scores)
        for i in range(n):
            scores[s[i]] = i / (n - 1)

    elif score_transform_type == 2 or score_transform_type == 3:
        # fitness shaping from "Natural Evolution Strategies" (Wierstra 2014) paper, either with zero mean (2) or without (3)
        lmbda = len(scores)
        s = np.argsort(-scores)
        for i in range(lmbda):
            scores[s[i]] = i + 1
        scores = scores.astype(float)
        for i in range(lmbda):
            scores[i] = max(0, np.log(lmbda / 2 + 1) - np.log(scores[i]))

        scores = scores / sum(scores)

        if score_transform_type == 2:
            scores -= 1 / lmbda

        scores /= max(scores)

    elif score_transform_type == 4:
        # consider single best eps
        scores_tmp = np.zeros(scores.size)
        scores_tmp[np.argmax(scores)] = 1
        scores = scores_tmp
    elif score_transform_type == 5:
        # consider all eps that are better than the average
        avg_score_orig = np.mean(scores_orig)

        scores_idx = np.where(scores > avg_score_orig + 1e-6, 1, 0)  # 1e-6 to counter numerical errors
        if sum(scores_idx) > 0:
            # if sum(scores_idx) > 0:
            scores = scores_idx * (scores - avg_score_orig) / (max(scores) - avg_score_orig + 1e-9)
            scores /= max(scores)
        else:
            scores = scores_idx

    elif score_transform_type == 6:
        # consider single best eps that is better than the average
        avg_score_orig = np.mean(scores_orig)

        scores_idx = np.where(scores > avg_score_orig + 1e-6, 1, 0)  # 1e-6 to counter numerical errors
        if sum(scores_idx) > 0:
            scores_tmp = np.zeros(scores.size)
            scores_tmp[np.argmax(scores)] = 1
            scores = scores_tmp
        else:
            scores = scores_idx
    else:
        raise ValueError("Unknown rank transform type: " + str(score_transform_type))

    score_transform_list = scores.tolist()

    return score_transform_list


def plot_score_transform_lists(score_list, score_orig_list, score_transform_lists):
    fix, axes = plt.subplots(nrows=1, ncols=len(score_transform_lists), dpi=600, figsize=(14, 3.5))

    titles = ['linear transf.', 'rank transf.', 'NES', 'NES unnorm.', 'single best', 'all better', 'single better']

    for i, score_transform_list in enumerate(score_transform_lists):
        axes[i].plot(score_list, linestyle='', marker='o')
        axes[i].plot([0, len(score_list)], [np.mean(score_orig_list), np.mean(score_orig_list)], color='r')
        axes[i].bar(range(len(score_transform_list)), score_transform_list, alpha=0.5)
        axes[i].set_ylim(-0.28, 1.05)
        axes[i].set_title(titles[i])

        axes[i].set_xlabel('population member')

        if i == 0:
            axes[i].set_ylabel('performance / fitness value')
            axes[i].legend([r'$\mathcal{R}_{G,i}$', r'$\mathcal{R}_G$', r'fitness value'])
        else:
            axes[i].set_yticks([])

    #plt.show()
    plt.savefig('visualize_rank_transform.eps')


    # fig = plt.figure(dpi=200, figsize=(5, 4))
    # plt.plot(score_list, linestyle='', marker='o')
    # plt.plot([0, len(score_list) - 1], [np.mean(score_orig_list), np.mean(score_orig_list)], color='r')
    #
    # for i, score_transform_list in enumerate(score_transform_lists):
    #     x = np.arange(len(score_transform_list)) + (i-3)/5
    #     plt.bar(x, score_transform_list, width=0.3, alpha=0.5)
    # plt.show()




if __name__ == "__main__":
    score_list = sorted([random.random()*0.3+0.6 for _ in range(20)])
    score_orig_list = [0.8]*20

    score_transform_lists = []

    for score_transform_type in range(7):
        score_transform_list = score_transform(score_list=score_list,
                                               score_orig_list=score_orig_list,
                                               score_transform_type=score_transform_type)
        score_transform_lists.append(score_transform_list)

    plot_score_transform_lists(score_list=score_list,
                               score_orig_list=score_orig_list,
                               score_transform_lists=score_transform_lists)










































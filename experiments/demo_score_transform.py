import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import random

def score_transform(score_list, score_orig_list, score_transform_type, nes_step_size):
    scores = np.asarray(score_list)
    scores_orig = np.asarray(score_orig_list)

    if score_transform_type == 0:
        # convert [1, 0, 5] to [0.2, 0, 1]
        scores = (scores - min(scores)) / (max(scores)-min(scores)+1e-9)

    elif score_transform_type == 1:
        # convert [1, 0, 5] to [0.5, 0, 1]
        s = np.argsort(scores)
        n = len(scores)
        for i in range(n):
            scores[s[i]] = i / (n-1)

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
        # consider single best eps that is better than the average
        avg_score_orig = np.mean(scores_orig)

        scores_idx = np.where(scores > avg_score_orig + 1e-6,1,0)   # 1e-6 to counter numerical errors
        if sum(scores_idx) > 0:
            scores_tmp = np.zeros(scores.size)
            scores_tmp[np.argmax(scores)] = 1
            scores = scores_tmp
        else:
            scores = scores_idx

    elif score_transform_type == 6 or score_transform_type == 7:
        # consider all eps that are better than the average, normalize weight sum to 1
        avg_score_orig = np.mean(scores_orig)

        scores_idx = np.where(scores > avg_score_orig + 1e-6,1,0)   # 1e-6 to counter numerical errors
        if sum(scores_idx) > 0:
        #if sum(scores_idx) > 0:
            scores = scores_idx * (scores-avg_score_orig) / (max(scores)-avg_score_orig+1e-9)
            if score_transform_type == 6:
                scores /= max(scores)
            else:
                scores /= sum(scores)
        else:
            scores = scores_idx

    else:
        raise ValueError("Unknown rank transform type: " + str(score_transform_type))

    if nes_step_size:
        scores = scores / len(score_list)

    score_transform_list = scores.tolist()

    return score_transform_list


def plot_score_transform_lists(score_list, score_orig_list, score_transform_lists, score_transform_lists_nes):
    fix, axes = plt.subplots(nrows=1, ncols=len(score_transform_lists), dpi=600, figsize=(10, 3.5))

    titles = ['linear transf.', 'rank transf.', 'NES', 'NES unnorm.', 'single best', 'single better', 'all better 1', 'all better 2']

    for i, lists in enumerate(zip(score_transform_lists, score_transform_lists_nes)):
        score_transform_list, score_transform_list_nes = lists
        axes[i].plot(score_list, linestyle='', marker='o')
        axes[i].plot([0, len(score_list)], [np.mean(score_orig_list), np.mean(score_orig_list)], color='r')
        axes[i].bar(range(len(score_transform_list)), score_transform_list, color='#78B0D7')
        #axes[i].bar(range(len(score_transform_list_nes)), score_transform_list_nes, color='#25587D')
        axes[i].set_ylim(-0.45, 1.05)
        axes[i].set_title(titles[i])


        if i == 0:
            axes[i].set_xlabel('i (population member)')
            axes[i].set_ylabel('expected cumulative reward / fitness value')
            #axes[i].legend([r'$K_i$', r'$K_G$', r'$F_i$', r'$F^\ast_i$'], loc='lower left')
            axes[i].legend([r'$K_i$', r'$K_G$', r'$F_i$'], loc='lower left')
        else:
            axes[i].set_yticks([])

    #plt.show()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('demo_score_transform.svg', bbox_inches='tight')


    # fig = plt.figure(dpi=200, figsize=(5, 4))
    # plt.plot(score_list, linestyle='', marker='o')
    # plt.plot([0, len(score_list) - 1], [np.mean(score_orig_list), np.mean(score_orig_list)], color='r')
    #
    # for i, score_transform_list in enumerate(score_transform_lists):
    #     x = np.arange(len(score_transform_list)) + (i-3)/5
    #     plt.bar(x, score_transform_list, width=0.3, alpha=0.5)
    # plt.show()




if __name__ == "__main__":
    score_list = sorted([random.random()*0.5+0.5 for _ in range(10)])
    score_orig_list = [0.85]*10

    score_transform_lists = []
    score_transform_lists_nes = []


    for score_transform_type in range(8):
        score_transform_list = score_transform(score_list=score_list,
                                               score_orig_list=score_orig_list,
                                               score_transform_type=score_transform_type,
                                               nes_step_size=False)
        score_transform_list_nes = score_transform(score_list=score_list,
                                                   score_orig_list=score_orig_list,
                                                   score_transform_type=score_transform_type,
                                                   nes_step_size=True)
        score_transform_lists.append(score_transform_list)
        score_transform_lists_nes.append(score_transform_list_nes)


    plot_score_transform_lists(score_list=score_list,
                               score_orig_list=score_orig_list,
                               score_transform_lists=score_transform_lists,
                               score_transform_lists_nes=score_transform_lists_nes)


import numpy as np
import time
from matplotlib import pyplot as plt


class BernoulliBandit(object):
    def __init__(self, prob):
        if prob is None:
            np.random.seed(int(time.time()))
            self.prob = np.random.random()
        else:
            self.prob = prob

    def generate_reward(self):
        if np.random.random() < self.prob:
            return 1
        else:
            return 0


class Bandit(object):
    def __init__(self, averageReward, variance):
        self.averageReward = averageReward
        self.variance = variance
        self.sigma = np.sqrt(self.variance)
        print('Bandit Created with avgReward {} and variance {}'.format(
            self.averageReward, self.variance))

    def generate_reward(self):
        return self.sigma * np.random.randn() + self.averageReward


def f1(a, b):
    if a == b:
        return 1
    else:
        return 0


def EpsilonGreedySolver(KArmedBandit, steps, epsilon, plot=True):
    K = len(KArmedBandit)

    q = [KArmedBandit[i].averageReward for i in range(K)]
    # epsilon greedy
    step = steps
    # epsilon = 0.1
    Q_n = [0 for i in range(K)]
    n = [0 for i in range(K)]
    r = 0
    t = 1
    # actionlist = []
    rewardlist = []
    avgRewardList = []
    # Start
    for i in range(step):
        if t % 1000 == 0:
            print(t)
            print(n)
            print(Q_n)
            print('#########################')
        u = np.random.random()
        if u < epsilon:
            # explore
            a = np.random.randint(0, K)             # randomly select in [0,K)
            # actionlist.append(a)                  # record
            n[a] += 1
            r = KArmedBandit[a].generate_reward()   # action applied
            rewardlist.append(r)  # record
            avgReward = sum(rewardlist) / len(rewardlist)
            avgRewardList.append(avgReward)
        else:
            a = Q_n.index(max(Q_n))                 # maxQ_Action
            # actionlist.append(a)                  # record
            n[a] += 1
            r = KArmedBandit[a].generate_reward()   # action applied
            rewardlist.append(r)                    # record
            avgReward = sum(rewardlist) / len(rewardlist)
            avgRewardList.append(avgReward)
        t += 1
        Q_n[a] = Q_n[a] + (1 / n[a]) * (r - Q_n[a])

        # print(Q_n)
        # print("#############################")

    print(Q_n)
    print(q)
    if plot is True:
        plt.plot([i for i in range(1, t)], avgRewardList)
    else:
        return avgRewardList


K = 10

avgReward = np.random.randn(K)
KArmedBandit = [Bandit(avgReward[i], 1) for i in range(K)]

steps = 10000
EpsilonGreedySolver(K, steps, 0, plot=True)
EpsilonGreedySolver(K, steps, 0.01, plot=True)
EpsilonGreedySolver(K, steps, 0.1, plot=True)
label = ["0", "0.01", '0.1']
plt.rc('legend', **{'fontsize': 12})
plt.legend(label, loc=0, ncol=2)
plt.grid()

plt.show()

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


def EpsilonGreedySolver(KArmedBandit, steps, epsilon):
    q = [KArmedBandit[i].averageReward for i in range(K)]
    # epsilon greedy
    step = steps
    # epsilon = 0.1
    Q = [0 for i in range(K)]
    r = 0
    t = 1
    actionlist = []
    rewardlist = []
    avgRewardList = []
    up = [[0 for i in range(step + 1)] for i in range(K)]
    down = [[0 for i in range(step + 1)] for i in range(K)]
    # Start
    for i in range(step):
        if t % 300 == 0:
            print(t)
        u = np.random.random()
        if u < epsilon:
            # explore
            a = np.random.randint(0, K)             # randomly select
            actionlist.append(a)                    # record
            r = KArmedBandit[a].generate_reward()   # action applied
            rewardlist.append(r)  # record
            up[a][t] = r
            down[a][t] = 1
            avgReward = sum(rewardlist) / len(rewardlist)
            avgRewardList.append(avgReward)
        else:
            a = Q.index(max(Q))                     # maxQ_Action
            actionlist.append(a)                    # record
            r = KArmedBandit[a].generate_reward()   # action applied
            rewardlist.append(r)                    # record
            up[a][t] = r
            down[a][t] = 1
            avgReward = sum(rewardlist) / len(rewardlist)
            avgRewardList.append(avgReward)
        t += 1
        for j in range(K):
            '''
            a, b = 0, 0
            for k in range(t):
                a += rewardlist[k] * f1(actionlist[k], j)
                b += f1(actionlist[k], j)
            '''
            a = sum(up[j])
            b = sum(down[j])
            if a == 0:
                Q[j] = 0
            else:
                Q[j] = a / b
        # print(Q)
        # print("#############################")

    print(Q)
    print(q)
    plt.plot([i for i in range(1, t)], avgRewardList)


K = 10
avgReward = np.random.randn(K)
KArmedBandit = [Bandit(avgReward[i], 1) for i in range(K)]

steps = 1000
EpsilonGreedySolver(KArmedBandit, steps, 0)
EpsilonGreedySolver(KArmedBandit, steps, 0.01)
EpsilonGreedySolver(KArmedBandit, steps, 0.1)
label = ["0", "0.01", '0.1']
plt.rc('legend', **{'fontsize': 12})
plt.legend(label, loc=0, ncol=2)
plt.grid()

plt.show()
'''
epsilon = 0.01
Q = [-0.43310759159746753, 0.4287883828576801, 0.25597582875112596, 0.5835792653180022, 1.3237091253158826,
     0.1266078951352569, -1.3507592834123265, -1.7785175555350707, 0.221190923697615, 1.0541365572540402]
q = [-0.43310759159746753, 0.42878838285767984, 0.25597582875112596, 0.5835792653180021, 1.323709125315981,
     0.12660789513525686, -1.3507592834123265, -1.7785175555350703, 0.221190923697615, 1.0541365572540278]
'''

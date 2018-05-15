import matplotlib
matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib

from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools
import WalkEnv.WalkEnv as Env

class OffPolicyTDn():
    def __init__(self, num_state, n, discount=0.9, step_size=0.8, target_policy=0.5, behavior_policy=0.6):
        self.num_state = num_state
        self.v_table = np.zeros(num_state)
        self.discount = discount
        self.step_size = step_size
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.v_true_value = self.calculate_true_V()
        self.n = n

    def update(self, sl, al, rl, d):
        # old_v_table = np.array(self.v_table)
        # n = len(al)
        s = sl[0]
        s_n = sl[-1]
        G = 0
        count = 0
        importance_sampling_ratio = 1.0
        for s, a, r in zip(sl[:-1], al, rl):
            G += pow(self.discount, count) * r
            importance_sampling_ratio *= self.target_policy / self.behavior_policy if a == 1 \
                else (1 - self.target_policy) / (1 - self.behavior_policy)
            count += 1

        if d:
            td_error = G - self.v_table[s_n]
        else:
            td_error = G + pow(self.discount, count) * self.v_table[s_n] - self.v_table[s]

        self.v_table[s] += self.step_size * importance_sampling_ratio * td_error

        RMS = np.sqrt(sum(np.square(self.v_true_value - self.v_table)))
#         print(old_v_table, self.v_table)
        return RMS

    def action(self):
        return np.random.choice(a=[0, 1], p=[1-self.behavior_policy, self.behavior_policy])

    def calculate_true_V(self):
        true_v = [i/(self.num_state-1) for i in range(self.num_state)]
        true_v[-1] = 0.0
        return np.array(true_v)


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


env = Env.RandomMove(live_display=True, len=19)

ITER = 100
THRESHOLD = 0.00001

alphas = [(a + 1)/10 for a in range(9)]
ns = [n + 1 for n in range(4)]
behavior_policy = 0.6

RMS_list = []

for a,n in itertools.product(alphas, ns):
    print('--for alpha:', a, 'and n:', n)
    offpolicy_tdn = OffPolicyTDn(num_state=env.n, behavior_policy=behavior_policy, step_size=a, n=n)
    RMSs = []
    steps = 0
    for i in range(ITER):
        iter_steps = 0
        # print('--Episode:', i)
        s = env.reset()
        sl = [s,]
        al =[]
        rl= []
        donel =[]
        while True:
            action = offpolicy_tdn.action()
            s_, r, d, _ = env.step(action)

            # a, r, s_, done
            al.append(action)
            rl.append(r)
            sl.append(s_)
            donel.append(d)

            iter_steps += 1
            steps += 1

            # TODO
            if iter_steps >= n:
                rms = offpolicy_tdn.update(sl=sl[-offpolicy_tdn.n-1:],
                                           al=al[-offpolicy_tdn.n:],
                                           rl=rl[-offpolicy_tdn.n:],
                                           d=donel[-1])
                                           # s_l=sl[-offpolicy_tdn.n:],
                                           # dl=donel[-offpolicy_tdn.n:])
                RMSs.append(rms)

            # update the last n-step
            if d:
                for k in range(1, offpolicy_tdn.n):
                    ind = offpolicy_tdn.n - k
                    rms = offpolicy_tdn.update(sl=sl[-ind-1:],
                                               al=al[-ind:],
                                               rl=rl[-ind:],
                                               d=donel[-1])
                    # s_l=sl[-offpolicy_tdn.n:],
                    # dl=donel[-offpolicy_tdn.n:])
                    RMSs.append(rms)
                break

            s = s_

    print('After', ITER, 'episodes.')
    print('Estimation of V values:', offpolicy_tdn.v_table.tolist(),
          'with target policy:', offpolicy_tdn.target_policy,
          'and behavior policy:', offpolicy_tdn.behavior_policy)
    print('RMS:', RMSs[-1:])
    RMS_list.append(RMSs[-1:])

rms_result = np.array(RMS_list).reshape((len(alphas), len(ns)))
print(rms_result)

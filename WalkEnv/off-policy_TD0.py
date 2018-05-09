import matplotlib
matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib

from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools
import WalkEnv as Env

class OffPolicyTD0():
    def __init__(self, num_state, discount=0.9, step_size=0.8, target_policy=0.5, behavior_policy=0.6):
        self.num_state = num_state
        self.v_table = np.zeros(num_state)
        self.discount = discount
        self.step_size = step_size
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.v_true_value = self.calculate_true_V()
        # print(self.v_true_value)

    def update(self, s, a, r, s_, d):
        # old_v_table = np.array(self.v_table)
        importance_sampling_ratio = self.target_policy / self.behavior_policy if a == 1 \
            else (1 - self.target_policy) / (1 - self.behavior_policy)
        if d:
            td_error = r - self.v_table[s]
        else:
            td_error = r + self.discount * self.v_table[s_] - self.v_table[s]
        self.v_table[s] += self.step_size * importance_sampling_ratio * td_error

        RMS = np.sqrt(sum(np.square(self.v_true_value - self.v_table))/self.num_state)
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

ITER = 1000
THRESHOLD = 0.00001

alphas = [(a + 1)/50 for a in range(10)]
behaviors = [0.7]
# behaviors = [(b + 1)/10 for b in range(9)]

RMS_list = []

for a,b in itertools.product(alphas,behaviors):
    print('--for alpha:', a, 'and behavior policy:', b)
    offpolicy_td0 = OffPolicyTD0(num_state=env.n, behavior_policy=b, step_size=a)
    RMSs = []
    steps = 0
    for i in range(ITER):
        # print('--Episode:', i)
        s = env.reset()
        while True:
            action = offpolicy_td0.action()
            s_, r, d, _ = env.step(action)
            rms = offpolicy_td0.update(s=s, a=action, r=r, s_=s_, d=d)
            RMSs.append(rms)
            if d:
                break
            s = s_
            steps += 1

    print('After', ITER, 'episodes.')
    print('Estimation of V values:', offpolicy_td0.v_table.tolist(),
          'with target policy:', offpolicy_td0.target_policy,
          'and behavior policy:', offpolicy_td0.behavior_policy)
    print('RMS:', RMSs[-1:])
    RMS_list.append(RMSs[-1:])

rms_result = np.array(RMS_list).reshape((len(alphas), len(behaviors)))
print(rms_result)

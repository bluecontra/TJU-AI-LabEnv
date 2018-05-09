import matplotlib
matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib

from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import WalkEnv as Env

env = Env.RandomMove(live_display=True)

env.reset()
obs = env._obs()
plt.imshow(obs, cmap=env.cmap, norm=env.norm)
plt.show()

for i in range(100):
    action = np.random.choice([0, 1])
    s, r, d, _ = env.step(action)
    obs = env._obs()
    plt.imshow(obs, cmap=env.cmap, norm=env.norm)
    plt.show()
    print(s, r, d)
    if d: break


# env = Env.RandomMove()
# for i in range(100):
#     action = np.random.choice([0, 1])
#     s, r, d, _ = env.step(action)
#     env.render()
#     if d: break
#     HTML(env._get_video(interval=200, gif_path='./try_env.gif').to_html5_video())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import json
from glob import glob


max_episode=int(1e5)


def load_record(record_file, scores_list, episodes_list):
    with open(record_file) as f:
        record = json.load(f)
        scores = np.array(record['evolving_scores'])
        scores_list.append(scores)
        episodes = np.array(record['evolving_episodes'])
        episodes_list.append(episodes)
        assert len(scores) == len(episodes)


def compute(scores_list, episodes_list):
    assert len(scores_list) == len(episodes_list)
    all_episodes = np.zeros(0, dtype=np.int32)
    for i in range(len(scores_list)):
        episodes = episodes_list[i]
        all_episodes = np.append(all_episodes, episodes)
        all_episodes = np.unique(all_episodes)
    if all_episodes[-1] < max_episode:
        all_episodes = np.append(all_episodes, max_episode)
    new_scores_array = np.zeros((len(scores_list), all_episodes.size), dtype=np.float32)
    for i in range(len(scores_list)):
        scores = scores_list[i]
        scores = np.insert(scores, 0, 0.)
        episodes = episodes_list[i]
        episodes = np.insert(episodes, 0, 0)
        episodes = np.append(episodes, all_episodes[-1]+1) 
        new_scores = np.zeros(all_episodes.size, dtype=np.float32)
        cursor = 1        
        for j in range(new_scores.size):            
            if all_episodes[j] >= episodes[cursor]:
                cursor += 1
            new_scores[j] = scores[cursor-1]
        new_scores_array[i, :] = new_scores
    score_std = np.std(new_scores_array, axis=0)
    score_mean = np.mean(new_scores_array, axis=0)
    return all_episodes, score_mean, score_std                                     


def draw(record_path, color_index, label):
    scores_list, episodes_list = [], []
    file_names = glob(record_path, recursive=True)
    for file_name in file_names:
        load_record(file_name, scores_list, episodes_list)
    all_episodes, score_mean, score_std = compute(scores_list, episodes_list)
    color = palette(color_index)
    plt.plot(all_episodes, score_mean, color=color, label=label, linewidth=3.5)
    plt.fill_between(all_episodes, score_mean-score_std, score_mean+score_std, color=color, alpha=0.2)


plt.style.use('seaborn-v0_8-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 32,
         }


fig = plt.figure(figsize=(20, 10))


# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1106/lr0.1-clip0.5/*/*.json', 4, 'ppo-median-lr0.1')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1106/lr0.3-clip0.5/*/*.json', 3, 'ppo-median-lr0.3')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1106/lr0.03-clip0.5/*/*.json', 2, 'ppo-median-lr0.03-clip0.5')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1102/median-lr0.01-clip0.2/*/*.json', 5, 'ppo-median-lr0.01-clip0.2')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1102/median-lr0.01/*/*.json', 4, 'ppo-median-lr0.01')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1102/median-lr0.003/*/*.json', 3, 'ppo-median-lr0.003')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1102/mean-lr0.0003/*/*.json', 2, 'ppo-mean-lr0.0003')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1102/median-lr0.0003/*/*.json', 3, 'ppo-median-lr0.0003')
# draw('/mnt/robustlearning/fitting/outputs/ppo/2025-1102/median-lr0.001/*/*.json', 2, 'ppo-median-lr0.001')

# draw('/mnt/robustlearning/fitting/outputs/ppo/saltpepper_noise/0.6/1/noisy_1/2025-1109/*/*.json', 2, 'PPO')
# draw('/mnt/robustlearning/fitting/outputs/cuckoo/saltpepper_noise/0.6/1/noisy_1/2025-1109/*/*.json', 1, 'CS')

draw('/mnt/robustlearning/fitting/outputs/ppo/saltpepper_noise/0.6/3/noisy_1/2025-1220/*/*.json', 2, 'PPO')
draw('/mnt/robustlearning/fitting/outputs/cuckoo/saltpepper_noise/0.6/3/noisy_1/2026-0129/*/*.json', 1, 'CS')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Number of fitness evaluations', fontsize=32)
plt.ylabel('Fitness', fontsize=32)
plt.legend(loc='lower right', prop=font1)
# plt.title("instance", fontsize=34)
fig.savefig('comparison.pdf')
plt.show()

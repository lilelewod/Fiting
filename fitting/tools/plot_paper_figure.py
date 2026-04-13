import numpy as np
import matplotlib.pyplot as plt
import json
import os

max_episode = int(1e5)


def load_record(record_file, scores_list, episodes_list):
    with open(record_file, 'r', encoding='utf-8') as f:
        record = json.load(f)
        scores = np.array(record['evolving_scores'])
        scores_list.append(scores)
        episodes = np.array(record['evolving_episodes'])
        episodes_list.append(episodes)


def compute(scores_list, episodes_list):
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
        episodes = np.append(episodes, all_episodes[-1] + 1)
        new_scores = np.zeros(all_episodes.size, dtype=np.float32)
        cursor = 1
        for j in range(new_scores.size):
            if all_episodes[j] >= episodes[cursor]:
                cursor += 1
            new_scores[j] = scores[cursor - 1]
        new_scores_array[i, :] = new_scores

    score_std = np.std(new_scores_array, axis=0)
    score_mean = np.mean(new_scores_array, axis=0)
    return all_episodes, score_mean, score_std


def draw(base_dir, color, label):
    scores_list, episodes_list = [], []

    # 强制获取绝对路径，确保不管在哪运行都不会找错
    abs_base_dir = os.path.abspath(base_dir)
    print(f"正在目录 {abs_base_dir} 中搜索 [{label}] 的数据...")

    if os.path.exists(abs_base_dir):
        for root, dirs, files in os.walk(abs_base_dir):
            if 'record.json' in files:
                file_path = os.path.join(root, 'record.json')
                try:
                    load_record(file_path, scores_list, episodes_list)
                    print(f"  -> 成功读取: {file_path}")
                except Exception as e:
                    print(f"  -> 失败: {file_path} - {e}")
    else:
        print(f" 错误: 路径不存在 - {abs_base_dir}")
        return

    if len(scores_list) == 0:
        print(f"警告: 没有读取到 [{label}] 的有效数据，跳过绘制。")
        return

    all_episodes, score_mean, score_std = compute(scores_list, episodes_list)

    # 绘制论文级别的带阴影平滑曲线
    plt.plot(all_episodes, score_mean, color=color, label=label, linewidth=2.0)
    plt.fill_between(all_episodes, score_mean - score_std, score_mean + score_std, color=color, alpha=0.2)


if __name__ == "__main__":
    # 设置论文风格的全局字体和网格
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['figure.facecolor'] = 'white'

    fig = plt.figure(figsize=(10, 6))

    # 获取当前脚本的绝对路径 (tools 文件夹)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上一级，获取项目的根目录 (fitting 文件夹)
    project_root = os.path.dirname(current_script_dir)

    # 拼出 outputs 下的 cs 和 cco 文件夹路径
    cs_dir = os.path.join(project_root, 'outputs', 'cs')
    cco_dir = os.path.join(project_root, 'outputs', 'cco')

    draw(cs_dir, color='#1f77b4', label='CS')  # 蓝色，对应原图的 CS
    draw(cco_dir, color='#2ca02c', label='CCO')  # 绿色，替代原图的 PPO

    plt.xlabel('Number of fitness evaluations', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 设置 x 轴显示范围到 100,000
    plt.xlim(0, 100000)

    # 获取图例并放置在右下角
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc='lower right', fontsize=12, frameon=True)

    # 保存为高清图片
    output_filename = os.path.join(current_script_dir, 'paper_figure_d.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n 已保存到: {output_filename}")
    plt.show()
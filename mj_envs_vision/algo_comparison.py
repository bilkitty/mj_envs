import os
import sys
import torch
import json
import numpy as np
import pickle as pkl
from PIL import Image
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs



BASE_PATH = "/home/bilkit/Workspace/mj_envs_vision/results/"
color_cycle = [plt.cm.get_cmap('Pastel2')(x) for x in range(10)]


def plot_rewards(ax, rewards: List[Tuple], label: str, color: Tuple[float], yaxis_label="total reward"):
  ep = np.array([x[0] for x in rewards])
  rwd = np.array([x[1] for x in rewards])
  if len(rwd.shape) == 1:
    ax.plot(ep, rwd, linestyle='solid', linewidth=2.5, label=label, color=color)
    rwd = rwd.reshape(1, -1)
  else:
    mu, std, med = np.mean(rwd, axis=-1), np.std(rwd, axis=-1), np.median(rwd, axis=-1)
    ax.plot(ep, mu, linestyle='dashed', linewidth=2.3, color=color)
    ax.plot(ep, med, linestyle='solid', linewidth=2.5, label=label, color=color)
    ax.fill_between(ep, mu - std, mu + std, alpha=0.25, color=color)

  ax.set_xlabel('epochs')
  ax.set_ylabel(f'{yaxis_label} n=({rwd.shape[-1]})')
  ax.legend(loc='upper right')
  return ax


def plot_metrics(fig, m_gs, metrics: dict[str, list], label: str, color: Tuple[float]):
  n = len(metrics.items())

  ax_gs = m_gs.subgridspec((n // 4 if (n % 4 == 0) else n // 4 + 1), 4)
  (labels, yvals) = list(map(list, zip(*metrics.items())))
  for i in range(n):
    y = np.array(yvals[i]).reshape(-1, 1)
    ep = np.linspace(0, len(y), len(y))
    ax = fig.add_subplot(ax_gs[i])

    ax.plot(ep, y, linestyle='dashed', linewidth=2.3, color=color, label=labels[i])
    ax.set_xlabel('epochs')
    ax.set_ylabel(f'{labels[i]} n=({y.shape[-1]})')
    #ax.legend(loc='upper right')
    ax.set_box_aspect(1)

    if i == 0:
      ax.set_title(f"{label} metrics")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage:\n\talgo_comparison.py <path_to_plot_config>")
    sys.exit(-1)

  with open(sys.argv[1], 'r') as fp:
    runs = json.load(fp)

  out_dir = runs["out_dir"] + '/compare_' + '_'.join(runs["run_names"])
  print('\033[96m' + f"saving results to {out_dir}" + '\033[0m')
  os.makedirs(out_dir, exist_ok=True)

  fig1 = plt.figure(figsize=tuple(runs["d_figure"]))
  grids = gs.GridSpec(len(runs["run_names"]), 1, figure=fig1, height_ratios=runs["proportions"])
  for i, exp in enumerate(runs["train_metrics_comparison"]):
    metrics = pkl.load(open(os.path.join(BASE_PATH, exp), 'rb'))
    plot_metrics(fig1, grids[i], metrics, label=runs["run_names"][i], color=color_cycle[i%10])

  fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
  for i, exp in enumerate(runs["train_comparison"]):
    rewards = pkl.load(open(os.path.join(BASE_PATH, exp), 'rb'))
    plot_rewards(ax2, rewards, label=runs["run_names"][i], color=color_cycle[i%10], yaxis_label="train reward")

  fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
  for i, exp in enumerate(runs["eval_comparison"]):
    rewards = pkl.load(open(os.path.join(BASE_PATH, exp), 'rb'))
    plot_rewards(ax3, rewards, label=runs["run_names"][i], color=color_cycle[i%10], yaxis_label="eval reward")

  fig1.tight_layout(pad=0.2)
  fig2.tight_layout(pad=0.2)
  fig3.tight_layout(pad=0.2)

  fig1.savefig(os.path.join(out_dir, "train_metrics.png"))
  fig2.savefig(os.path.join(out_dir, "train_reward.png"))
  fig3.savefig(os.path.join(out_dir, "eval_reward.png"))


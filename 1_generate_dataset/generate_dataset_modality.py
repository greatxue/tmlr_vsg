import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import argparse

# 布局算法字典
layout_algorithms = {
    'spring': nx.spring_layout,
    'circular': nx.circular_layout,
    'kamada_kawai': nx.kamada_kawai_layout,
    'random': nx.random_layout,
    'shell': nx.shell_layout,
    'spectral': nx.spectral_layout
}

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--layout', type=str, choices=layout_algorithms.keys(), default='spring')
args = parser.parse_args()

# 加载 TUDataset 数据集
dataset_name = args.dataset
dataset = TUDataset(root='./raw_dataset', name=dataset_name)

# 创建保存图像和图数据结构的文件夹
output_dir = f"./modality/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)

# 图像可视化参数
node_size = 200
edge_width = 2
figsize = (8, 8)
dpi = 100

def visualize_graph(graph, index, label, layout_algorithm=nx.spring_layout):
    """
    可视化并保存图的布局，同时保存图数据结构。

    参数：
    - graph: NetworkX 图对象
    - index: 图的索引
    - label: 图的标签
    - layout_algorithm: 布局算法（默认为 nx.spring_layout）
    """
    pos = layout_algorithm(graph, dim=2)
    plt.figure(figsize=figsize, facecolor='black')
    nx.draw(graph, pos, node_size=node_size, node_color='gray', with_labels=False, node_shape='s')
    nx.draw_networkx_edges(graph, pos, edge_color='w', width=edge_width)
    plt.tight_layout()

    # 保存图像文件
    image_path = f"{output_dir}/graph_{index}.png"
    plt.savefig(image_path, format='png', dpi=dpi, facecolor='black')
    plt.close()

    return image_path

# 保存图像和图数据结构的列表
image_paths = []
labels = []
graph_data_paths = []

# 遍历数据集中的每个图，并保存其图像和图数据结构
for idx, data in enumerate(dataset):
    # 创建 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())

    # 可视化并保存图像
    image_path = visualize_graph(G, index=idx, label=data.y.item(), layout_algorithm=layout_algorithms[args.layout])
    image_paths.append(image_path)
    labels.append(data.y.item())

    # 保存图数据结构
    graph_data_path = f"{output_dir}/graph_{idx}.pkl"
    with open(graph_data_path, 'wb') as f:
        pickle.dump(data, f)
    graph_data_paths.append(graph_data_path)

# 创建包含图像路径、图数据路径和标签的数据集 CSV 文件
data = {
    'image_path': image_paths,
    'graph_data_path': graph_data_paths,
    'label': labels
}
df = pd.DataFrame(data)
df.to_csv(f"{output_dir}/dataset.csv", index=False)

print("图像和图数据结构保存完成，并生成了包含路径和标签的 CSV 文件！")
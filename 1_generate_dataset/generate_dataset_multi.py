import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse

# 定义不同的布局算法
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
args = parser.parse_args()

datasetname = args.dataset
dataset = TUDataset(root='./raw_dataset', name=datasetname)

# 创建保存图像的文件夹
output_dir = f"./img/{datasetname}_multi"
os.makedirs(output_dir, exist_ok=True)

# 增加节点大小和边宽度
node_size = 200
edge_width = 2
figsize = (8, 8)  # 图像的宽度和高度（以英寸为单位）
dpi = 100  # 每英寸像素数

def visualize_graph(graph, index, label, layout_name, layout_algorithm):
    """
    可视化并保存图的布局。

    参数：
    - graph: NetworkX图对象
    - index: 图的索引
    - label: 图的标签
    - layout_name: 布局算法名称
    - layout_algorithm: 布局算法函数
    """
    pos = layout_algorithm(graph, dim=2)
    plt.figure(figsize=figsize, facecolor='black') 
    nx.draw(graph, pos, node_size=node_size, node_color='gray', with_labels=False, node_shape='s')
    nx.draw_networkx_edges(graph, pos, edge_color='w', width=edge_width)
    plt.tight_layout()
    
    # 保存图像到对应文件夹
    folder_path = f"{output_dir}/graph_{index}"
    os.makedirs(folder_path, exist_ok=True)
    image_path = f"{folder_path}/graph_{index}_{layout_name}.png"
    plt.savefig(image_path, format='png', dpi=dpi, facecolor='black')
    plt.close()
    return image_path

# 保存图像和标签的列表
image_paths = []
labels = []

# 遍历数据集中的每个图，并保存其不同视角的图像
for idx, data in enumerate(dataset):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())
    
    # 为每张图生成不同布局的图像
    for layout_name, layout_algorithm in layout_algorithms.items():
        image_path = visualize_graph(G, index=idx, label=data.y.item(), layout_name=layout_name, layout_algorithm=layout_algorithm)
        image_paths.append(image_path)
        labels.append(data.y.item())

# 创建一个包含图像路径和标签的数据集
data = {'image_path': image_paths, 'label': labels}
df = pd.DataFrame(data)
df.to_csv(f"{output_dir}/dataset.csv", index=False)

print("图像保存完成，并生成了包含图像路径和标签的数据集！")
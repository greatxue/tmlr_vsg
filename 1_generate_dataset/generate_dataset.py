import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse

# 加载MUTAG数据集
layout_algorithms = {
    'spring': nx.spring_layout,
    'circular': nx.circular_layout,
    'kamada_kawai': nx.kamada_kawai_layout,
    'random': nx.random_layout,
    'shell': nx.shell_layout,
    'spectral': nx.spectral_layout
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--layout', type=str, choices=layout_algorithms.keys(), default='spring')
args = parser.parse_args()

datasetname = args.dataset
dataset = TUDataset(root='./raw_dataset', name=datasetname)

# 创建保存图像的文件夹
output_dir = f"./img/{datasetname}"
os.makedirs(output_dir, exist_ok=True)

# 增加节点大小和边宽度
node_size = 200
edge_width = 2
figsize = (8, 8)  # 图像的宽度和高度（以英寸为单位）
dpi = 100  # 每英寸像素数

def visualize_graph(graph, index, label, layout_algorithm=nx.spring_layout):
    """
    可视化并保存图的布局。

    参数：
    - graph: NetworkX图对象
    - index: 图的索引
    - label: 图的标签
    - layout_algorithm: 布局算法（默认为nx.spring_layout）
    """
    pos = layout_algorithm(graph, dim=2)
    plt.figure(figsize=figsize, facecolor='black') 
    nx.draw(graph, pos, node_size=node_size, node_color='gray', with_labels=False,node_shape='s')
    nx.draw_networkx_edges(graph, pos, edge_color='w', width=edge_width)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph_{index}.png", format='png', dpi=dpi, facecolor='black')
    plt.close()

# 保存图像和标签的列表
image_paths = []
labels = []

# 遍历数据集中的每个图，并保存其图像
for idx, data in enumerate(dataset):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())
    visualize_graph(G, index=idx, label=data.y.item(),layout_algorithm=layout_algorithms[args.layout])
    image_paths.append(f"{output_dir}/graph_{idx}.png")
    labels.append(data.y.item())

# 创建一个包含图像路径和标签的数据集
data = {'image_path': image_paths, 'label': labels}
df = pd.DataFrame(data)
df.to_csv(f"{output_dir}/dataset.csv", index=False)

print("图像保存完成，并生成了包含图像路径和标签的数据集！")






"""上一版代码：

import os
import os.path as osp
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torch_geometric.datasets import TUDataset

# 设置数据集名称和路径
dataset_name = 'PROTEINS'
dataset_path = f'./img_m/{dataset_name}/'
num_images_per_sample = 5  # 每个样本生成的图像数量

# 创建数据集目录
if not osp.exists(dataset_path):
    os.makedirs(dataset_path)

# 生成带有节点特征和染色的图像
def create_colored_graph_image(graph, node_features, edge_features, output_path, node_size=300, edge_width=2.0):
    if node_features is not None:
        if np.issubdtype(node_features[0][0].dtype, np.integer):  # 判断是否是离散特征
            node_colors = node_features.flatten()
            num_colors = len(np.unique(node_colors))
            norm = Normalize(vmin=node_colors.min(), vmax=node_colors.max())
            cmap = plt.cm.get_cmap('tab20', num_colors)
        else:  # 连续特征
            kmeans = KMeans(n_clusters=2)
            node_colors = kmeans.fit_predict(node_features)
            norm = Normalize(vmin=node_colors.min(), vmax=node_colors.max())
            cmap = plt.cm.rainbow
    else:
        node_colors = 'blue'
        norm = None
        cmap = None

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 8))
    
    # 动态调整节点大小和边宽度
    node_size = 200
    edge_width = 2   
    # 绘制节点，颜色根据特征分组（如果有特征）
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_size, cmap=cmap)
    
    # 绘制边，粗细根据特征（如果有边特征）
    if edge_features is not None:
        edges = graph.edges()
        edge_weights = [edge_features[edge] for edge in edges]
        nx.draw_networkx_edges(graph, pos, edge_color=edge_weights, edge_cmap=plt.cm.Blues, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights), width=edge_width)
    else:
        nx.draw_networkx_edges(graph, pos, width=edge_width)
    
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

# 加载TUDataset
dataset = TUDataset(root='./raw_dataset', name=dataset_name)

with open(osp.join(dataset_path, 'dataset.csv'), 'w') as f:
    f.write('image_path,label\n')
    
    # 遍历数据集
    for i, data in enumerate(dataset):
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.t().tolist())
        
        # 使用节点特征，如果没有节点特征则生成随机特征
        if data.x is not None:
            node_features = data.x.numpy()
            print('################', data.x[0][0].dtype)
        else: 
            node_features = None
        
        # 使用边特征，如果没有边特征则设置为None
        edge_features = None
        if data.edge_attr is not None:
            edge_features = {(data.edge_index[0, j].item(), data.edge_index[1, j].item()): data.edge_attr[j].item() for j in range(data.edge_index.shape[1])}
        
        for j in range(num_images_per_sample):
            # 设置图像路径
            image_path = f'graph_{i}_{j}.png'
            full_image_path = osp.join(dataset_path, image_path)
            
            # 生成图像
            create_colored_graph_image(G, node_features, edge_features, full_image_path)
            
            # 写入CSV文件
            f.write(f'{image_path},{data.y.item()}\n')

print("数据集制作完成！")
"""


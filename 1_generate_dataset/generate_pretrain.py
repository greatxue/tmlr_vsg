import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# 布局算法
layout_algorithms = {
    'spring': nx.spring_layout,
    'circular': nx.circular_layout,
    'kamada_kawai': nx.kamada_kawai_layout,
    'random': nx.random_layout,
    'shell': nx.shell_layout,
    'spectral': nx.spectral_layout
}

# 可视化参数
node_size = 200
edge_width = 2
figsize = (8, 8)
dpi = 100

def visualize_graph(graph, output_path, layout_algorithm=nx.spring_layout):
    """可视化并保存图"""
    pos = layout_algorithm(graph, dim=2)
    plt.figure(figsize=figsize, facecolor='black')
    nx.draw(graph, pos, node_size=node_size, node_color='gray', 
            with_labels=False, node_shape='s', arrows=True)
    nx.draw_networkx_edges(graph, pos, edge_color='w', width=edge_width, 
                           arrowsize=10)
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=dpi, facecolor='black')
    plt.close()

def generate_chain_graphs(min_nodes=2, max_nodes=50, num_graphs=100):
    """生成链式图"""
    return [nx.path_graph(random.randint(min_nodes, max_nodes)) 
            for _ in range(num_graphs)]

def generate_multi_cycle_graphs(min_nodes=10, max_nodes=50, min_cycles=1, max_cycles=5, num_graphs=500):
    """生成多个循环的图，确保没有孤立节点"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # 确保所有节点都被使用
        unused_nodes = set(range(n))
        num_cycles = random.randint(min_cycles, max_cycles)
        
        # 生成第一个循环，确保至少有一个较大的循环
        first_cycle_length = random.randint(max(3, n//3), min(n, n//2))
        first_cycle_nodes = random.sample(list(unused_nodes), first_cycle_length)
        G.add_edges_from(zip(first_cycle_nodes, first_cycle_nodes[1:] + [first_cycle_nodes[0]]))
        unused_nodes -= set(first_cycle_nodes)
        
        # 生成剩余的循环
        for _ in range(num_cycles - 1):
            if len(unused_nodes) < 3:  # 如果剩余节点不足以形成新的环，就连接到现有环
                break
                
            # 确定这个循环的长度
            max_possible_length = min(len(unused_nodes), n//3)
            if max_possible_length < 3:
                break
            cycle_length = random.randint(3, max_possible_length)
            
            # 选择节点，包括至少一个已使用的节点来确保连通性
            used_node = random.choice(list(set(range(n)) - unused_nodes))
            cycle_nodes = [used_node] + random.sample(list(unused_nodes), cycle_length-1)
            
            # 添加环
            G.add_edges_from(zip(cycle_nodes, cycle_nodes[1:] + [cycle_nodes[0]]))
            unused_nodes -= set(cycle_nodes[1:])  # 更新未使用节点集合
        
        # 如果还有未使用的节点，将它们连接到图中
        while unused_nodes:
            node = unused_nodes.pop()
            # 随机连接到已使用的节点
            connected_node = random.choice(list(set(range(n)) - unused_nodes))
            G.add_edge(node, connected_node)
        
        graphs.append(G)
    return graphs

def generate_ego_graphs(min_nodes=10, max_nodes=30, min_k=2, max_k=6, min_p=0.1, max_p=0.5, min_radius=1, max_radius=3, num_graphs=100):
    """生成更加随机化的ego网络"""
    graphs = []
    for _ in range(num_graphs):
        # 随机选择节点数量
        n = random.randint(min_nodes, max_nodes)
        
        # 随机选择每个节点的邻居数 k
        k = random.randint(min_k, min(max_k, n-1))
        
        # 随机选择重连概率 p
        p = random.uniform(min_p, max_p)
        
        # 生成Watts-Strogatz小世界网络
        G = nx.watts_strogatz_graph(n, k, p)
        
        # 随机选择中心节点
        center = random.randint(0, n-1)
        
        # 随机选择ego网络的半径
        radius = random.randint(min_radius, max_radius)
        
        # 提取ego网络
        ego_graph = nx.ego_graph(G, center, radius=radius)
        
        # 随机删除一些边（使网络更稀疏）
        edges = list(ego_graph.edges())
        num_edges_to_remove = random.randint(0, len(edges) // 3)  # 最多删除1/3的边
        edges_to_remove = random.sample(edges, num_edges_to_remove)
        ego_graph.remove_edges_from(edges_to_remove)
        
        graphs.append(ego_graph)
    return graphs

def generate_er_graphs(min_nodes=10, max_nodes=30, p=0.3, num_graphs=100):
    """生成ER随机图"""
    p = random.uniform(0.1, p)
    return [nx.erdos_renyi_graph(random.randint(min_nodes, max_nodes), p) 
            for _ in range(num_graphs)]

def generate_community_graphs(min_nodes=20, max_nodes=40, num_communities=3, p_in=0.1, p_out=0.005, num_graphs=500):
    """生成具有社区结构的图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        sizes = [n // num_communities] * num_communities
        sizes[-1] += n % num_communities  # 处理余数
        G = nx.random_partition_graph(sizes, p_in, p_out)
        graphs.append(G)
    return graphs

def generate_tree_graphs(min_nodes=5, max_nodes=20, num_graphs=50):
    """生成随机树"""
    return [nx.random_tree(random.randint(min_nodes, max_nodes)) 
            for _ in range(num_graphs)]

def generate_bipartite_graphs(min_nodes=5, max_nodes=15, p=0.3, num_graphs=50):
    """生成随机二部图"""
    return [nx.bipartite.random_graph(
        random.randint(min_nodes, max_nodes), 
        random.randint(min_nodes, max_nodes), p) 
        for _ in range(num_graphs)]

def generate_regular_graphs(min_nodes=6, max_nodes=20, d=3, num_graphs=50):
    """生成d-正则图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        if n > d and (n*d) % 2 == 0:
            try:
                G = nx.random_regular_graph(d, n)
                graphs.append(G)
            except nx.NetworkXError:
                continue
    return graphs

def generate_graphs_with_cut_vertices(min_nodes=10, max_nodes=50, num_graphs=100):
    """生成具有割点的图"""
    return [nx.barbell_graph(random.randint(min_nodes//2, max_nodes//2), 1) 
            for _ in range(num_graphs)]

def generate_graphs_with_bridges(min_nodes=10, max_nodes=50, num_graphs=100):
    """生成具有桥的图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        G = nx.path_graph(n)
        for _ in range(n):
            u, v = random.sample(range(n), 2)
            if not nx.has_path(G, u, v) or len(nx.shortest_path(G, u, v)) > 2:
                G.add_edge(u, v)
        graphs.append(G)
    return graphs

def generate_planar_graphs(min_nodes=10, max_nodes=20, num_graphs=50):
    """生成平面图"""
    return [nx.planar_graph(random.randint(min_nodes, max_nodes)) 
            for _ in range(num_graphs)]

def generate_directed_graphs_with_scc(min_nodes=10, max_nodes=50, num_graphs=100):
    """生成带有强连通分量的有向图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        G = nx.strongly_connected_components_subgraphs(
            nx.random_k_out_graph(n, 3, 0.5)
        )
        G = nx.compose_all(G)
        graphs.append(G)
    return graphs

def generate_graphs_with_self_loops(min_nodes=10, max_nodes=50, num_graphs=100):
    """生成带环的图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        G = nx.erdos_renyi_graph(n, 0.3)
        for _ in range(n // 3):
            node = random.choice(list(G.nodes()))
            G.add_edge(node, node)
        graphs.append(G)
    return graphs

def generate_graphs_with_leaves(min_nodes=10, max_nodes=20, num_graphs=50):
    """生成带有叶子节点的图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        G = nx.random_tree(n)
        leaves = [node for node in G.nodes() if G.degree(node) == 1]
        for _ in range(n // 3):
            u, v = random.sample([node for node in G.nodes() if node not in leaves], 2)
            G.add_edge(u, v)
        graphs.append(G)
    return graphs

def generate_complete_multipartite_graphs(min_parts=3, max_parts=5, min_nodes_per_part=2, max_nodes_per_part=5, num_graphs=50):
    """生成完全k部图"""
    graphs = []
    for _ in range(num_graphs):
        k = random.randint(min_parts, max_parts)
        sizes = [random.randint(min_nodes_per_part, max_nodes_per_part) for _ in range(k)]
        G = nx.complete_multipartite_graph(*sizes)
        graphs.append(G)
    return graphs

def generate_graphs_with_isolated_nodes(min_nodes=10, max_nodes=20, num_graphs=50):
    """生成带有孤立节点的图"""
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        G = nx.erdos_renyi_graph(n, 0.3)
        for _ in range(n // 5):
            G.add_node(G.number_of_nodes())
        graphs.append(G)
    return graphs

def main():
    base_dir = "./pretrain"
    graph_types = {
        'chain': generate_chain_graphs,
        'multi_cycle': generate_multi_cycle_graphs,
        'ego': generate_ego_graphs,
        'er': generate_er_graphs,
        'community': generate_community_graphs,
        'tree': generate_tree_graphs,
        'bipartite': generate_bipartite_graphs,
        'regular': generate_regular_graphs,
        'cut_vertices': generate_graphs_with_cut_vertices,
        'bridges': generate_graphs_with_bridges,
        # 'planar': generate_planar_graphs,
        # 'directed_scc': generate_directed_graphs_with_scc,
        'self_loops': generate_graphs_with_self_loops,
        'leaves': generate_graphs_with_leaves,
        'multipartite': generate_complete_multipartite_graphs,
        'isolated': generate_graphs_with_isolated_nodes
    }
    
    all_image_paths = []
    all_labels = []
    
    for graph_type, generator_func in graph_types.items():
        type_dir = os.path.join(base_dir, graph_type)
        os.makedirs(type_dir, exist_ok=True)
        
        graphs = generator_func()
        
        for i, G in enumerate(graphs):
            output_path = os.path.join(type_dir, f"graph_{i}.png")
            visualize_graph(G, output_path)
            all_image_paths.append(output_path)
            all_labels.append(graph_type)
    
    data = {'image_path': all_image_paths, 'label': all_labels}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "dataset.csv"), index=False)
    
    print("图像生成完成，并生成了包含图像路径和标签的数据集！")

if __name__ == "__main__":
    main()
from nilearn import datasets
import networkx as nx
import pickle

# 注释掉的是不方便计算评估的参数
def cal_graph(fc, threshold):
    atlas_labels = datasets.fetch_atlas_aal()['labels']
    # 创建图
    fcg = nx.Graph()
    fcg.add_edges_from([(atlas_labels[i], atlas_labels[j]) for i in range(fc.shape[0]) for j in range(fc.shape[1]) if i != j and abs(fc[i][j]) >= threshold/100])
    fcg.add_nodes_from(atlas_labels)
    if len(fcg.edges) == 0:
        print(f"图为空")
        return
    this_run = {}
    this_run["graph"] = fcg

    # # degree（度）
    # this_run["degree"] = fcg.degree

    # k-core（k度核）
    this_run["k_core"] = max(nx.core_number(nx.k_core(fcg)).values())

    # # 中心性
    # # degree centrality（度中心性）
    # this_run["degree_centrality"] = nx.degree_centrality(fcg)

    # # closeness centrality（接近中心性）
    # this_run["closeness_centrality"] = nx.closeness_centrality(fcg)

    # # betweenness centrality（中介中心性）
    # this_run["betweenness_centrality"] = nx.betweenness_centrality(fcg)

    # try:
    #     # Eigenvector Centrality（特征向量中心性）
    #     this_run["eigenvector_centrality"] = nx.eigenvector_centrality(fcg)
    # except Exception as e:
    #     print(e)

    # 聚类性质
    # # numbers of triangles（三角形数）
    # this_run["triangles"] = nx.triangles(fcg)

    # clustering efficiency（聚类系数）
    this_run["clustering"] = nx.clustering(fcg)

    # transitivity（传递性）
    this_run["transitivity"] = nx.transitivity(fcg)

    # 关联性
    # Network Assortativity（网络关联性）
    this_run["degree_assortativity_coefficient"] = nx.degree_assortativity_coefficient(fcg)

    # # average neighbor degree（平均邻居度）
    # this_run["average_neighbor_degree"] = nx.average_neighbor_degree(fcg)

    # # average degree connectivity（平均度连接）
    # this_run["average_degree_connectivity"] = nx.average_degree_connectivity(fcg)

    # 效率性质
    # global efficiency（全局效率）
    this_run["global_efficiency"] = nx.global_efficiency(fcg)

    # local efficiency（局部效率）
    this_run["local_efficiency"] = nx.local_efficiency(fcg)

    # ratio of local to global efficiency（局部全局效率比）
    this_run["ratio"] = this_run["local_efficiency"]/this_run["global_efficiency"]

    # try:
    #     # characteristic path length/average shortest path length（特征路径长度/平均最短路径长度）
    #     this_run["char_path_len"] = nx.average_shortest_path_length(fcg)
    #     this_run["connected"] = True

    #     # # small-world coefficient（小世界系数）
    #     # this_run["small_world"] = nx.sigma(fcg, 30)
    # except nx.NetworkXError as e:
    #     this_run["connected"] = False
    #     print(e)
    
    return this_run

def multi_process(dFC, sub, threshold, save_path):
    graph = {}
    graph["ses-preop"] = {}
    for run, dfc in dFC["ses-preop"].items():
        graph["ses-preop"][run] = []
        for fc in dfc:
            fcg = cal_graph(fc, threshold)
            if fcg is not None:
                graph["ses-preop"][run].append(fcg)
    graph["ses-postop"] = {}
    for run, dfc in dFC["ses-postop"].items():
        graph["ses-postop"][run] = []
        for fc in dfc:
            fcg = cal_graph(fc, threshold)
            if fcg is not None:
                graph["ses-postop"][run].append(fcg)
    print(sub, "计算完成")
    with open(f"{save_path}/{sub}.pkl", "wb") as f:
        pickle.dump(graph, f)
    print(sub, "保存成功")
    return graph, sub
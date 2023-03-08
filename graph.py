# %%
import pickle
import numpy as np
from nilearn import plotting, datasets
import networkx as nx
import os

# %%
atlas_labels = datasets.fetch_atlas_aal()['labels']
with open("FCs.pkl", "rb") as f:
    fcs:dict = pickle.load(f)

# %%
save_path = "graph_theory"
os.makedirs(save_path, exist_ok=True)
for sub in fcs:
    fc_graph_theory= {}
    for ses in fcs[sub]:
        fc_graph_theory[ses] = {}
        for run, fc in fcs[sub][ses].items():
            fc_graph_theory[ses][run] = {}
            this_run = fc_graph_theory[ses][run]

            # %%
            # 创建图
            fcg = nx.Graph()
            fcg.add_edges_from([(atlas_labels[i], atlas_labels[j]) for i in range(fc.shape[0]) for j in range(fc.shape[1]) if i != j and fc[i][j] >= 0.5])
            this_run["graph"] = fcg

            # %%
            # degree（度）
            this_run["degree"] = fcg.degree

            # k-core（k度核）
            this_run["main_core"] = nx.k_core(fcg)

            # %%
            # 中心性
            # degree centrality（度中心性）
            this_run["degree_centrality"] = nx.degree_centrality(fcg)

            # closeness centrality（接近中心性）
            this_run["closeness_centrality"] = nx.closeness_centrality(fcg)

            # betweenness centrality（中介中心性）
            this_run["betweenness_centrality"] = nx.betweenness_centrality(fcg)

            # Eigenvector Centrality（特征向量中心性）
            this_run["eigenvector_centrality"] = nx.eigenvector_centrality(fcg)

            # %%
            # 聚类性质
            # numbers of triangles（三角形数）
            this_run["triangles"] = nx.triangles(fcg)

            # clustering efficiency（聚类系数）
            this_run["clustering"] = nx.clustering(fcg)

            # transitivity（传递性）
            this_run["transitivity"] = nx.transitivity(fcg)

            # %%
            # 关联性
            # Network Assortativity（网络关联性）
            this_run["degree_assortativity_coefficient"] = nx.degree_assortativity_coefficient(fcg)

            # average neighbor degree（平均邻居度）
            this_run["average_neighbor_degree"] = nx.average_neighbor_degree(fcg)

            # average degree connectivity（平均度连接）
            this_run["average_degree_connectivity"] = nx.average_degree_connectivity(fcg)

            # %%
            # 效率性质
            # global efficiency（全局效率）
            this_run["global_efficency"] = nx.global_efficiency(fcg)

            # local efficiency（局部效率）
            this_run["local_efficency"] = nx.local_efficiency(fcg)

            # ratio of local to global efficiency（局部全局效率比）
            this_run["ratio"] = this_run["local_efficency"]/this_run["global_efficency"]

            # %%
            try:
                # rich club
                rich_club = nx.rich_club_coefficient(fcg)
            except nx.NetworkXError as e:
                print(e)

            try:
                # characteristic path length/average shortest path length（特征路径长度/平均最短路径长度）
                this_run["char_path_len"] = nx.average_shortest_path_length(fcg)

                # small-world coefficient（小世界系数）
                this_run["small_world"] = nx.sigma(fcg, 30)
            except nx.NetworkXError as e:
                print(f"{e} : {sub}-{ses}-{run}")

            # %%
            with open(f"{save_path}/{sub}.pkl", "wb") as f:
                pickle.dump(fc_graph_theory, f)

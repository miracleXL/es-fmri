from nilearn import connectome, plotting
import numpy as np
from sklearn import cluster, metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import os

mpl.use("agg")

def sliceWindows(time_series, frame, interval):
    if len(time_series) < frame:
        return [time_series]
    windows = []
    l, r = 0, frame
    while r < time_series.shape[0]:
        windows.append(time_series[l:r])
        l += interval
        r = l + frame
    return windows

def plot_sates(states, save_path=None):
    x = [0] * len(states) * 2
    y = [0] * len(states) * 2
    for i in range(1, len(states)):
        x[2*i-1] = x[2*i] = i
        y[2*i] = y[2*i+1] = states[i]
    x[-1] = len(states)
    y[0] = y[1] = states[0]

    # x = list(range(1, len(states)+1))
    # y = states

    fig, ax = plt.subplots(figsize=(20, 5))
    fig.patch.set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(15, integer=True))
    ax.plot(x,y)
    if save_path:
        fig.savefig(save_path, format="png")
    plt.cla()
    plt.clf()
    plt.close("all")

def clustering_evaluate(windows, ks, save_path):
    fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)

    inertias = []
    scs = []
    chs = []
    dbs = []
    fcs2d = fcs.reshape((fcs.shape[0], 13456))
    for k in ks:
        if k < fcs2d.shape[0]:
            center, states, inertia = cluster.k_means(fcs2d, k)
            inertias.append(inertia) # 肘点法
            scs.append(metrics.silhouette_score(fcs2d, states)) # 轮廓系数
            chs.append(metrics.calinski_harabasz_score(fcs2d, states)) # CH，方差比
            dbs.append(metrics.davies_bouldin_score(fcs2d, states)) # DB
        else:
            inertias.append(inertias[-1])
            scs.append(scs[-1])
            chs.append(chs[-1])
            dbs.append(dbs[-1])
    # 绘图
    figi, axi = plt.subplots(2, 2, figsize=(20, 10))
    figi.patch.set_color("white")
    axi[0, 0].set_title("elbow method")
    axi[0, 0].plot(ks, inertias)
    axi[0, 1].set_title("Silhouette Coefficient")
    axi[0, 1].plot(ks, scs)
    axi[1, 0].set_title("calinski harabasz")
    axi[1, 0].plot(ks, chs)
    axi[1, 1].set_title("davies bouldin")
    axi[1, 1].plot(ks, dbs)
    figi.savefig(save_path, format="png")
    plt.cla()
    plt.clf()
    plt.close("all")

def windows_evaluate(data, subid, window_lengths, step, k, save_path):

    inertias = []
    scs = []
    chs = []
    dbs = []
    for time in window_lengths:
        windows_preop = []
        for run, items in data[subid]["ses-preop"].items():
            stepTR = math.ceil(step/items["TR"])
            preopTR = math.ceil(time/items["TR"])
            windows_preop += sliceWindows(items["time_series"], preopTR, stepTR)
        # windows_postop = []
        # for run, items in data[subid]["ses-postop"].items():
        #     stepTR = math.ceil(step/items["TR"])
        #     windows_postop += sliceWindows(items["time_series"], time/items["TR"], stepTR)
        fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows_preop)
        fcs2d = fcs.reshape((fcs.shape[0], 13456))
        if k < fcs2d.shape[0]:
            center, states, inertia = cluster.k_means(fcs2d, k)
            inertias.append(inertia) # 肘点法
            scs.append(metrics.silhouette_score(fcs2d, states)) # 轮廓系数
            chs.append(metrics.calinski_harabasz_score(fcs2d, states)) # CH，方差比
            dbs.append(metrics.davies_bouldin_score(fcs2d, states)) # DB
        else:
            inertias.append(inertias[-1])
            scs.append(scs[-1])
            chs.append(chs[-1])
            dbs.append(dbs[-1])
    # 绘图
    figi, axi = plt.subplots(2, 2, figsize=(20, 10))
    figi.patch.set_color("white")
    axi[0, 0].set_title("elbow method")
    axi[0, 0].plot(window_lengths, inertias)
    axi[0, 1].set_title("Silhouette Coefficient")
    axi[0, 1].plot(window_lengths, scs)
    axi[1, 0].set_title("calinski harabasz")
    axi[1, 0].plot(window_lengths, chs)
    axi[1, 1].set_title("davies bouldin")
    axi[1, 1].plot(window_lengths, dbs)
    os.makedirs(f"{save_path}/{k}states", exist_ok=True)
    figi.savefig(f"{save_path}/{k}states/{step}.png", format="png")
    plt.cla()
    plt.clf()
    plt.close("all")

def align(time_series:np.ndarray, length:int, start_point=None):
    """
    start_point: None, "middle", "end", "random"
    """
    if time_series.shape[0] <= length:
        return time_series
    if start_point == "middle":
        sp = (time_series.shape[0]-length)//2
        return time_series[sp:sp+length]
    elif start_point == "end":
        return time_series[-length:]
    elif start_point == "random":
        sp = np.random.randint(time_series.shape[0]-length)
        return time_series[sp:sp+length]
    return time_series[:length]

def cluster_save_states(windows_preop, windows_postop, k, save_path):
    fcs_preop = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows_preop)
    fcs_postop = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows_postop)
    fcs2d_preop = fcs_preop.reshape((fcs_preop.shape[0], 13456))
    fcs2d_postop = fcs_postop.reshape((fcs_postop.shape[0], 13456))
    km = cluster.KMeans(k)
    states_preop = km.fit_predict(fcs2d_preop)
    states_postop = km.predict(fcs2d_postop)
    fit_states_postop = km.fit_predict(fcs2d_postop)
    plot_sates(states_preop, f"{save_path}/preop.png")
    plot_sates(states_postop, f"{save_path}/postop.png")
    plot_sates(fit_states_postop, f"{save_path}/fit_postop.png")

def save_fcs(fcs:list, save_path:str, names:list=None):
    fcs.sort(key=lambda fc:fc.sum())
    for run, fc in enumerate(fcs):
        np.fill_diagonal(fc, 0)
        fig_fc, ax_fc = plt.subplots()
        fc_display = plotting.plot_matrix(fc, figure=fig_fc, colorbar=True, vmax=0.8, vmin=-0.8)
        fig_fc.patch.set_color("white")
        fig_fc.savefig(f"{save_path}/{names[run] if names else run}.png", format="png")
    plt.cla()
    plt.clf()
    plt.close("all")

def save_fc(fc:np.ndarray, save_path:str):
    np.fill_diagonal(fc, 0)
    fig_fc, ax_fc = plt.subplots()
    fc_display = plotting.plot_matrix(fc, figure=fig_fc, colorbar=True, vmax=0.8, vmin=-0.8)
    fig_fc.patch.set_color("white")
    fig_fc.savefig(save_path, format="png")
    plt.cla()
    plt.clf()
    plt.close("all")

def states2fcs(states:list, km:cluster.KMeans):
    return [km.cluster_centers_[state].reshape((116,116)) for state in states]

def get_missing_state(windows_preop, windows_postop, k, save_path):
    fcs_preop = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows_preop)
    fcs_postop = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows_postop)
    fcs2d_preop = fcs_preop.reshape((fcs_preop.shape[0], 13456))
    fcs2d_postop = fcs_postop.reshape((fcs_postop.shape[0], 13456))
    km = cluster.KMeans(k)
    states_preop = km.fit_predict(fcs2d_preop)
    states_postop = km.predict(fcs2d_postop)
    fit_states_postop = km.fit_predict(fcs2d_postop)
    plot_sates(states_preop, f"{save_path}/preop.png")
    plot_sates(states_postop, f"{save_path}/postop.png")
    plot_sates(fit_states_postop, f"{save_path}/fit_postop.png")
    state = set(states_preop)^set(states_postop)
    if not state:
        print(state, set(states_preop), set(states_postop))
        return
    state = state.pop()
    fc = km.cluster_centers_[state].reshape((116,116))
    # save_fc(fc, f"{save_path}/missing.png")
    save_fcs([fc.reshape((116,116)) for fc in km.cluster_centers_], save_path)
    return fcs_preop[states_preop == state]
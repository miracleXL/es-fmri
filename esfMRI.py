from nilearn import connectome, plotting, datasets
import numpy as np
from sklearn import cluster, metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import math
import os

atlas_labels = datasets.fetch_atlas_aal()['labels']

def sliceWindows(time_series, frame, interval):
    """ 切割滑动窗口 """
    """
        time_series: numpy.ndarray, 需要切割的时间序列
        frame: int, 每帧窗口尺寸
        interval: int, 窗口间步长间隔
    """
    if len(time_series) < frame:
        return [time_series]
    windows = []
    l, r = 0, frame
    while r < time_series.shape[0]:
        windows.append(time_series[l:r])
        l += interval
        r = l + frame
    return windows

def AIC(n, k, L):
    """
    n:数据数量; k: 模型参数(状态数); L: 残差平方和
    AIC = 2k+ n*ln(L/n)
    """
    return 2*k + n*math.log(L/n)

def BIC(n, k, L):
    """ BIC(C) = n*ln(L/n) + k*ln(n) """
    return n*math.log(L/n) + k*math.log(n)

def HQ(n, k, L):
    """ HQ = -2*ln(L) + ln(ln(n))*k """
    return -2*math.log(L) + math.log(math.log(n))*k

def plot_sates(states, k, save_path=None):
    """
    绘制状态变化过程
    states: list[int], 所有需要绘制出来的状态列表，每个状态应该在[0, k)之间
    k: int, 总状态数
    save_path: str, 保存路径
    """
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
    ax.set_yticks(range(k))
    ax.plot(x,y)
    if save_path:
        fig.savefig(save_path, format="png")
    plt.cla()
    plt.clf()
    plt.close("all")

def plot_evaluated(x_axis, inertias=None, scs=None, chs=None, dbs=None, aic=None, bic=None, save_path=None):
    figi, axi = plt.subplots(3, 2, figsize=(20, 10))
    figi.patch.set_color("white")
    if inertias is not None:
        axi[0, 0].set_title("elbow method")
        axi[0, 0].plot(x_axis, inertias)
    if scs is not None:
        axi[0, 1].set_title("Silhouette Coefficient")
        axi[0, 1].plot(x_axis, scs)
    if chs is not None:
        axi[1, 0].set_title("calinski harabasz")
        axi[1, 0].plot(x_axis, chs)
    if dbs is not None:
        axi[1, 1].set_title("davies bouldin")
        axi[1, 1].plot(x_axis, dbs)
    if aic is not None:
        axi[2, 0].set_title("AIC")
        axi[2, 0].plot(x_axis, aic)
    if bic is not None:
        axi[2, 1].set_title("BIC")
        axi[2, 1].plot(x_axis, bic)
    if save_path is not None:
        figi.savefig(save_path, format="png")
        plt.cla()
        plt.clf()
        plt.close("all")


def clustering_mean_shift(windows):
    fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
    fcs2d = fcs.reshape((fcs.shape[0], 13456))
    ms = cluster.MeanShift()
    states = ms.fit_predict(fcs2d)
    return ms, states

def clustering_evaluate(windows, ks, save_path):
    fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)

    inertias = []
    scs = []
    chs = []
    dbs = []
    aic = []
    bic = []
    fcs2d = fcs.reshape((fcs.shape[0], 13456))
    for k in ks:
        if k < fcs2d.shape[0]:
            center, states, inertia = cluster.k_means(fcs2d, k)
            inertias.append(inertia) # 肘点法
            scs.append(metrics.silhouette_score(fcs2d, states)) # 轮廓系数
            chs.append(metrics.calinski_harabasz_score(fcs2d, states)) # CH，方差比
            dbs.append(metrics.davies_bouldin_score(fcs2d, states)) # DB
            aic.append(AIC(fcs2d.shape[0], k, inertia))
            bic.append(BIC(fcs2d.shape[0], k, inertia))
        else:
            inertias.append(inertias[-1])
            scs.append(scs[-1])
            chs.append(chs[-1])
            dbs.append(dbs[-1])
            aic.append(aic[-1])
            bic.append(bic[-1])
    # 绘图
    plot_evaluated(ks, inertias=inertias, scs=scs, chs=chs, dbs=dbs, aic=aic, bic=bic, save_path=save_path)

def windows_evaluate(data, subid, window_lengths, step, k, save_path):
    """
    评估窗口尺寸对聚类效果的影响。
    data: dict, 所有数据的字典
    subid: str, 待评估被试id
    window_lengths: list, 所有待测试窗口尺寸
    step: int, 窗口步长
    k: int, 目标状态数
    save_path: str, 保存路径
    """

    inertias = []
    scs = []
    chs = []
    dbs = []
    aic = []
    bic = []
    for time in window_lengths:
        windows_preop = []
        for run, items in data[subid]["ses-preop"].items():
            stepTR = math.ceil(step/items["TR"])
            preopTR = math.ceil(time/items["TR"])
            windows_preop += sliceWindows(items["time_series"], preopTR, stepTR)
        windows_postop = []
        for run, items in data[subid]["ses-postop"].items():
            stepTR = math.ceil(step/items["TR"])
            postopTR = math.ceil(time/items["TR"])
            windows_postop += sliceWindows(items["time_series"], postopTR, stepTR)
        windows = windows_preop + windows_postop
        fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
        fcs2d = fcs.reshape((fcs.shape[0], 13456))
        if k < fcs2d.shape[0]:
            center, states, inertia = cluster.k_means(fcs2d, k)
            inertias.append(inertia) # 肘点法
            scs.append(metrics.silhouette_score(fcs2d, states)) # 轮廓系数
            chs.append(metrics.calinski_harabasz_score(fcs2d, states)) # CH，方差比
            dbs.append(metrics.davies_bouldin_score(fcs2d, states)) # DB
            aic.append(AIC(fcs2d.shape[0], k, inertia))
            bic.append(BIC(fcs2d.shape[0], k, inertia))
        else:
            inertias.append(inertias[-1])
            scs.append(scs[-1])
            chs.append(chs[-1])
            dbs.append(dbs[-1])
            aic.append(aic[-1])
            bic.append(bic[-1])
    # 绘图
    plot_evaluated(window_lengths, inertias=inertias, scs=scs, chs=chs, dbs=dbs, aic=aic, bic=bic, save_path=save_path)

def step_evaluate(window_length, steps, k, save_path, time_series_preop=None, time_series_postop=None):
    if time_series_preop is None and time_series_postop is None:
        return
    inertias = {"preop":[], "postop":[], "total":[]}
    scs = {"preop":[], "postop":[], "total":[]}
    chs = {"preop":[], "postop":[], "total":[]}
    dbs = {"preop":[], "postop":[], "total":[]}
    aic = {"preop":[], "postop":[], "total":[]}
    bic = {"preop":[], "postop":[], "total":[]}
    for step in steps:
        if time_series_preop is not None:
            windows = []
            for run, items in time_series_preop.items():
                preopTR = math.ceil(window_length/items["TR"])
                windows += sliceWindows(items["time_series"], preopTR, step)
            fcs_preop = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
            fcs2d_preop = fcs_preop.reshape((fcs_preop.shape[0], 13456))
            if k < fcs2d_preop.shape[0]:
                center, states, inertia = cluster.k_means(fcs2d_preop, k)
                inertias["preop"].append(inertia) # 肘点法
                scs["preop"].append(metrics.silhouette_score(fcs2d_preop, states)) # 轮廓系数
                chs["preop"].append(metrics.calinski_harabasz_score(fcs2d_preop, states)) # CH，方差比
                dbs["preop"].append(metrics.davies_bouldin_score(fcs2d_preop, states)) # DB
                aic["preop"].append(AIC(fcs2d_preop.shape[0], k, inertia))
                bic["preop"].append(BIC(fcs2d_preop.shape[0], k, inertia))
            else:
                inertias["preop"].append(0)
                scs["preop"].append(0)
                chs["preop"].append(0)
                dbs["preop"].append(0)
                aic["preop"].append(0)
                bic["preop"].append(0)
        if time_series_postop is not None:
            windows = []
            for run, items in time_series_postop.items():
                postopTR = math.ceil(window_length/items["TR"])
                windows += sliceWindows(items["time_series"], postopTR, step)
            fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
            del windows
            fcs = fcs.reshape((fcs.shape[0], 13456))
            if k < fcs.shape[0]:
                center, states, inertia = cluster.k_means(fcs, k)
                inertias["postop"].append(inertia) # 肘点法
                scs["postop"].append(metrics.silhouette_score(fcs, states)) # 轮廓系数
                chs["postop"].append(metrics.calinski_harabasz_score(fcs, states)) # CH，方差比
                dbs["postop"].append(metrics.davies_bouldin_score(fcs, states)) # DB
                aic["postop"].append(AIC(fcs.shape[0], k, inertia))
                bic["postop"].append(BIC(fcs.shape[0], k, inertia))
            else:
                inertias["postop"].append(0)
                scs["postop"].append(0)
                chs["postop"].append(0)
                dbs["postop"].append(0)
                aic["postop"].append(0)
                bic["postop"].append(0)
        if time_series_preop is not None and time_series_postop is not None:
            fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
            fcs = fcs.reshape((fcs.shape[0], 13456))
            if k < fcs.shape[0]:
                center, states, inertia = cluster.k_means(fcs, k)
                inertias["total"].append(inertia) # 肘点法
                scs["total"].append(metrics.silhouette_score(fcs, states)) # 轮廓系数
                chs["total"].append(metrics.calinski_harabasz_score(fcs, states)) # CH，方差比
                dbs["total"].append(metrics.davies_bouldin_score(fcs, states)) # DB
                aic["total"].append(AIC(fcs.shape[0], k, inertia))
                bic["total"].append(BIC(fcs.shape[0], k, inertia))
            else:
                inertias["total"].append(0)
                scs["total"].append(0)
                chs["total"].append(0)
                dbs["total"].append(0)
                aic["total"].append(0)
                bic["total"].append(0)
    # 绘图
    if time_series_preop is not None:
        plot_evaluated(steps, inertias=inertias["preop"], scs=scs["preop"], chs=chs["preop"], dbs=dbs["preop"], aic=aic["preop"], bic=bic["preop"], save_path=f"{save_path}/preop_{window_length}_{k}_state.png")
    if time_series_postop is not None:
        plot_evaluated(steps, inertias=inertias["postop"], scs=scs["postop"], chs=chs["postop"], dbs=dbs["postop"], aic=aic["postop"], bic=bic["postop"], save_path=f"{save_path}/postop_{window_length}_{k}_state.png")
    if time_series_preop is not None and time_series_postop is not None:
        plot_evaluated(steps, inertias=inertias["total"], scs=scs["total"], chs=chs["total"], dbs=dbs["total"], aic=aic["total"], bic=bic["total"], save_path=f"{save_path}/total_{window_length}_{k}_state.png")

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

def cluster_save_states(windows, k, save_path):
    """ 聚类并保存状态 """
    fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
    fcs2d = fcs.reshape((fcs.shape[0], 13456))
    km = cluster.KMeans(k)
    states_preop = km.fit_predict(fcs2d)
    plot_sates(states_preop, k, f"{save_path}/preop.png")

def joint_cluster_save_states(windows_preop, windows_postop, k, save_path=None):
    """ 拼接电刺激前后数据，并聚类后保存 """
    windows = windows_preop + windows_postop
    fcs = connectome.ConnectivityMeasure(kind="correlation").fit_transform(windows)
    fcs2d = fcs.reshape((fcs.shape[0], 13456))
    km = cluster.KMeans(k)
    states = km.fit_predict(fcs2d)
    preop = len(windows_preop)
    states_preop = states[:preop]
    states_postop = states[preop:]
    if save_path is not None:
        plot_sates(states_preop, k, f"{save_path}/preop.png")
        plot_sates(states_postop, k, f"{save_path}/postop.png")
        save_fcs([fc.reshape((116,116)) for fc in km.cluster_centers_], save_path)
    return [states_preop, states_postop]

def average_time_series(data:dict, align_length):
    time_series = None
    count = 0
    for run, items in data.items():
        if items["time_series"].shape[0] < align_length:
            continue
        count += 1
        time_series = align(items["time_series"], align_length) if time_series is None else time_series + align(items["time_series"], align_length)
    if time_series is None:
        return None
    return time_series/count

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
    plot_sates(states_preop, k, f"{save_path}/preop.png")
    plot_sates(states_postop, k, f"{save_path}/postop.png")
    plot_sates(fit_states_postop, k, f"{save_path}/fit_postop.png")
    states = set(states_preop)^set(states_postop)
    if not states:
        return
    state = states.pop()
    fc = km.cluster_centers_[state].reshape((116,116))
    # save_fc(fc, f"{save_path}/missing.png")
    names = list(range(k))
    names[state] = f"missing_state_{state}"
    while states:
        st = states.pop()
        names[st] = f"missing_state_{st}"
    save_fcs([fc.reshape((116,116)) for fc in km.cluster_centers_], save_path, names)
    return fcs_preop[states_preop == state]

def stats_tests(preop, postop, print_values=True):
    p_values = []
    # 参数检验
    # 正态性检验
    norm_stat1, norm_p_value1 = stats.shapiro(preop)
    if print_values:
        print("正态性检验preop: ", norm_p_value1)
    norm_stat2, norm_p_value2 = stats.shapiro(postop)
    if print_values:
        print("正态性检验postop: ", norm_p_value2)
    # 方差齐性检验
    l_stat, l_p_value = stats.levene(preop, postop)
    if print_values:
        print("方差齐性检验: ", l_p_value)
    if norm_p_value1 > 0.05 and norm_p_value2 > 0.05 and l_p_value > 0.05:
        # student T 检验
        t_stat, t_p_value = stats.ttest_ind(preop, postop)
        if print_values:
            print("t检验: ", t_stat, t_p_value)
        p_values.append(t_p_value)
        # ANOVA
        f_stat, f_p_value = stats.f_oneway(preop, postop)
        if print_values:
            print("ANOVA检验: ", f_stat, f_p_value)
        p_values.append(f_p_value)
    else:
        p_values.append(0)
        p_values.append(0)
    # 非参数检验
    # 不要求正态分布
    if print_values:
        print("非参数检验: ")
    # Wilcoxon秩和检验
    rs_stat, rs_p_value = stats.ranksums(preop, postop)
    if print_values:
        print("Wilcoxon秩和检验: ", rs_stat, rs_p_value)
    p_values.append(rs_p_value)
    # 曼-惠特尼U检验
    mw_stat, mw_p_value = stats.mannwhitneyu(preop, postop)
    if print_values:
        print("曼-惠特尼U检验: ", mw_stat, mw_p_value)
    p_values.append(mw_p_value)
    # ks检验
    ks_stat, ks_p_value = stats.ks_2samp(preop, postop)
    if print_values:
        print("ks检验: ", ks_stat, ks_p_value)
    p_values.append(ks_p_value)
    # kruskal 原假设是各样本服从的概率分布具有相同的中位数，原假设被拒绝意味着至少一个样本的概率分布的中位数不同于其他样本
    k_stat, k_p_value = stats.kruskal(preop, postop)
    if print_values:
        print("kruskal检验", k_stat, k_p_value)
    p_values.append(k_p_value)
    # Dunnett's t 检验
    # SNK q检验
    return np.array(p_values)

def Euler_distance(matrix1:np.ndarray, matrix2:np.ndarray) -> int:
    """
        计算两个矩阵的欧式距离
        matrix1, matrix2: np.ndarray, 二维矩阵
    """
    return np.sqrt(np.sum(np.square(matrix2 - matrix1)))

def Manhattan_distance(matrix1:np.ndarray, matrix2:np.ndarray) -> int:
    """
        计算两个矩阵的曼哈顿距离
        matrix1, matrix2: np.ndarray, 二维矩阵
    """
    return np.sum(np.abs(matrix2 - matrix1))

def plot_graph_measures(measures_preop:dict, measures_postop:dict, title:str, save_path:str) -> None:
    """
        绘制动态功能连接中图论参数随窗口变化图。
        measures_*: {run: []}图论参数
    """
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.patch.set_color("white")
    ax[0].set_title(f"{title} preop")
    ax[1].set_title(f"{title} postop")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_yticks([i/10 for i in range(0, 11, 2)])
    ax[1].set_yticks([i/10 for i in range(0, 11, 2)])
    x = 0
    for run, measures in measures_preop.items():
        ax[0].plot(range(x, x+len(measures)), measures)
        # x += len(measures)
    fig.savefig(save_path, format="png")
    x = 0
    for run, measures in measures_postop.items():
        ax[1].plot(range(x, x+len(measures)), measures)
        # x += len(measures)
    fig.savefig(save_path, format="png")
    plt.cla()
    plt.clf()
    plt.close("all")
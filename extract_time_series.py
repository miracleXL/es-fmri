from nilearn import maskers, datasets
from nilearn.interfaces.fmriprep import load_confounds_strategy
# pickle是一个将Python对象保存为文件的模块
import pickle

# 加载一个数据集，被试数量为2
data = datasets.fetch_development_fmri(n_subjects=1)

# 加载ALL脑图谱模板
# load AAL atlas template.
atlas = datasets.fetch_atlas_aal()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']

# 创建ROI遮罩，以便从每个ROI中提取时间序列
masker = maskers.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                                    memory='nilearn_cache', memory_level=1, verbose=5)
# 提取时间序列，data.func是原始数据，confounds是噪音
time_series = masker.fit_transform(data.func[0], confounds=data.confounds)

# 保存为pickle文件
with open("time_series_test.pkl", "wb") as f:
    pickle.dump(time_series, f)

# 读取pickle文件
with open("time_series_test.pkl", "rb") as f:
    time_series = pickle.load(f)
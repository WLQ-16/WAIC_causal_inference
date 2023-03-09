import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ylearn.estimator_model import DML4CATE
import csv
import random

# 1.混淆因子估计
# 根据因果发现程序（causal_graph.py）生成的因果图，本研究选取了潜在的confounders列表，为了减少样本选择性偏差的影响，通过随机选择confounders，
# 计算treatment=1或2时的ATE，并以此为根据，获取计算ce_1和ce_2因果效应的confounder_list

# 导入数据预处理后的训练集数据和测试集数据，需要先运行data_processing.py
train = pd.read_csv('train_input.csv')
test = pd.read_csv('test_input.csv')

V = ['V_2', 'V_9', 'V_10', 'V_14', 'V_15', 'V_24', 'V_19', 'V_30', 'V_28', 'V_31', 'V_36', 'V_7', 'V_39']  # 因果图中出现对Y有因果影响最多的几个因子
for epochs in range(100):  # 先跑100轮
    confounder_list = random.sample(V, 6)  # 此处从V中随机选取6个进行实验，可以设置其他数量
    print(confounder_list)
    # 以下工作与因果效应估计（casual_estimator.py）完全一致
    V_new = train[confounder_list + ['treatment'] + ['outcome']]
    dml = DML4CATE(cf_fold=1,
                   x_model=RandomForestClassifier(n_estimators=500,
                                                  criterion="entropy",
                                                  max_depth=250,
                                                  min_samples_leaf=4,
                                                  min_samples_split=3,
                                                  max_features=2),
                   y_model=RandomForestRegressor(n_estimators=500,
                                                 max_depth=250,
                                                 min_samples_leaf=4,
                                                 min_samples_split=3,
                                                 max_features=2),
                   is_discrete_treatment=True)
    dml.fit(data=V_new, outcome='outcome', treatment='treatment', covariate=confounder_list)
    ce_dml = dml.effect_nji(data=V_new, control=0)

    ce_dml_test = dml.effect_nji(data=test, control=0)
    ce_dml_train = ce_dml[:, :, 1:].reshape(-1, 2)
    ce_dml_all = np.concatenate([ce_dml_train, ce_dml_test[:, :, 1:].reshape(-1, 2)], axis=0)

    ce_result = ce_dml_all
    save = pd.DataFrame(ce_result)
    print('ce_1:', format(save[0].mean(), '.4f'), 'ce_2:', format(save[1].mean(), '.4f'))
    # 将实验的结果写入test_V.csv文件中以供分析
    with open("test_V.csv", "a") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        row = [confounder_list, format(save[0].mean(), '.4f'), format(save[1].mean(), '.4f')]
        writer.writerows([row])

# 2.dml参数调整实验
# 对几个主要的dml参数进行调整，进而选取结果最佳的参数组合
for n_estimator_list in (1000, 1500):
    for min_samples_leaf_list in (2, 4, 20, 50):
        for min_samples_split_list in (2, 40):
            for max_features_list in (1, 2, 8):
                dml = DML4CATE(cf_fold=1,
                               x_model=RandomForestClassifier(n_estimators=n_estimator_list,
                                                              criterion="entropy",
                                                              max_depth=100,
                                                              min_samples_leaf=min_samples_leaf_list,
                                                              min_samples_split=min_samples_split_list,
                                                              max_features=max_features_list,
                                                              random_state=1104),
                               y_model=RandomForestRegressor(n_estimators=n_estimator_list,
                                                             max_depth=100,
                                                             min_samples_leaf=min_samples_leaf_list,
                                                             min_samples_split=min_samples_split_list,
                                                             max_features=max_features_list,
                                                             random_state=1104),
                               is_discrete_treatment=True,
                               random_state=1104)
                dml.fit(data=V_new, outcome='outcome', treatment='treatment', covariate=confounder_list)
                ce_dml = dml.effect_nji(data=V_new, control=0)
                ce_dml_test = dml.effect_nji(data=test, control=0)
                ce_dml_train = ce_dml[:, :, 1:].reshape(-1, 2)
                ce_dml_all = np.concatenate([ce_dml_train, ce_dml_test[:, :, 1:].reshape(-1, 2)], axis=0)
                ce_result = ce_dml_all
                save=pd.DataFrame(ce_result)
                col1 = n_estimator_list
                # 将调参结果保存到csv文件中以供进一步分析
                with open("test_dml_parameters.csv", "a") as csvfile:
                    writer = csv.writer(csvfile, lineterminator='\n')
                    row = [n_estimator_list, min_samples_leaf_list, min_samples_split_list, max_features_list,
                           format(save[0].mean(), '.4f'), format(save[1].mean(), '.4f'), format(save[0].mean()+save[1].mean(), '.4f')]
                    writer.writerows([row])
                print('n_estimators:', n_estimator_list, ',min_samples_leaf:', min_samples_leaf_list, ',min_samples_split:',
                      min_samples_split_list, ',max_features:', max_features_list, 'ce_1:', format(save[0].mean(), '.4f'),
                      'ce_2:', format(save[1].mean(), '.4f'), 'ce_1+ce_2=:', format(save[0].mean()+save[1].mean(), '.4f'))

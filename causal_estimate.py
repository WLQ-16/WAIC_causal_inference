import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ylearn.estimator_model import DML4CATE

#导入数据预处理后的训练集数据和测试集数据，需要先运行data_processing.py,为了方便起见我们已经生成了预处理后的文件，可以直接运行并得到结果
train = pd.read_csv('dataset/train_input.csv')
test = pd.read_csv('dataset/test_input.csv')

#根据因果分析结果（见README文档），确定分析treatment=1时的confounders
confounder_list_ce_1 = ['V_2', 'V_7', 'V_28', 'V_25', 'V_14', 'V_15', 'V_24']
V_new = train[confounder_list_ce_1 + ['treatment'] + ['outcome']]

# 使用dml方法计算treatment=1时的因果效应
# 关于dml方法的原理和使用步骤详见：https://ylearn.readthedocs.io/zh_CN/latest/sub/est_model/dml.html
# 由于随机森林具有一定的随机性，生成的结果会有些许不同，误差在0.01以内
dml = DML4CATE(cf_fold=1,
               x_model=RandomForestClassifier(n_estimators=1500,
                                              criterion="entropy",
                                              max_depth=100,
                                              min_samples_leaf=3,
                                              min_samples_split=2,
                                              max_features=None,
                                              random_state=2022),
               y_model=RandomForestRegressor(n_estimators=1800,
                                             max_depth=100,
                                             min_samples_leaf=3,
                                             min_samples_split=2,
                                             max_features=None,
                                             random_state=2022),
               is_discrete_treatment=True,
               random_state=1104)
dml.fit(data=V_new, outcome='outcome', treatment='treatment', covariate=confounder_list_ce_1)
ce_dml = dml.effect_nji(data=V_new, control=0)
ce_dml_test = dml.effect_nji(data=test, control=0)
ce_dml_train = ce_dml[:, :, 1:].reshape(-1, 2)
ce_dml_all = np.concatenate([ce_dml_train, ce_dml_test[:, :, 1:].reshape(-1, 2)], axis=0)
ce_result = ce_dml_all
save_ce_1 = pd.DataFrame(ce_result)
print('ce_1:', format(save_ce_1[0].mean(), '.4f'))

# 使用dml方法计算treatment=2时的因果效应
confounder_list_ce_2 = ['V_2', 'V_28', 'V_24', 'V_9', 'V_10']
V_new = train[confounder_list_ce_2 + ['treatment'] + ['outcome']]
dml = DML4CATE(cf_fold=1,
               x_model=RandomForestClassifier(n_estimators=1500,
                                              criterion="entropy",
                                              max_depth=100,
                                              min_samples_leaf=10,
                                              min_samples_split=3,
                                              max_features=3,
                                              random_state=1104),
               y_model=RandomForestRegressor(n_estimators=1500,
                                             max_depth=100,
                                             min_samples_leaf=10,
                                             min_samples_split=3,
                                             max_features=3,
                                             random_state=1104),
               is_discrete_treatment=True,
               random_state=1104)
dml.fit(data=V_new, outcome='outcome', treatment='treatment', covariate=confounder_list_ce_2)
ce_dml = dml.effect_nji(data=V_new, control=0)
ce_dml_test = dml.effect_nji(data=test, control=0)
ce_dml_train = ce_dml[:, :, 1:].reshape(-1, 2)
ce_dml_all = np.concatenate([ce_dml_train, ce_dml_test[:, :, 1:].reshape(-1, 2)], axis=0)
ce_result = ce_dml_all
save_ce_2=pd.DataFrame(ce_result)
print('ce_2:', format(save_ce_2[1].mean(), '.4f'))

#合并对ce_1和ce_2的预测，输出最终结果
save = pd.concat([save_ce_1[0], save_ce_2[1]], axis=1)
print('ce_1:', format(save[0].mean(), '.4f'), 'ce_2:', format(save[1].mean(), '.4f'))
save.to_csv('result.csv', index=0, header=['ce_1', 'ce_2'])


import pandas as pd
import numpy as np

train = pd.read_csv('dataset/train.csv')#导入原始训练数据
test = pd.read_csv('dataset/test.csv')#导入原始测试数据
def build_data(train):
    # 字符串类型定性转换
    # V_8、V_10、V14、V_26中的yes赋值为1，no赋值为0
    train.loc[train['V_8'] == 'no', 'V_8'] = int(0)
    train.loc[train['V_8'] == 'yes', 'V_8'] = int(1)
    train.loc[train['V_10'] == 'no', 'V_10'] = int(0)
    train.loc[train['V_10'] == 'yes', 'V_10'] = int(1)
    train.loc[train['V_14'] == 'no', 'V_14'] = int(0)
    train.loc[train['V_14'] == 'yes', 'V_14'] = int(1)
    train.loc[train['V_26'] == 'no', 'V_26'] = int(0)
    train.loc[train['V_26'] == 'yes', 'V_26'] = int(1)

    # 缺失值赋值，方便后续的类型转换
    # 赋值原则：选取原始特征列中不存在的数值，以便缺失值/非缺失值的区分
    # 先暂时把V_10、V_14、V_26中的缺失值赋值为2，V_4赋值为0、V_28赋值为2800（赋值并不影响预测效果，只用作区分缺失值）
    train.loc[train['V_10'] == 'unknown', 'V_10'] = int(2)
    train.loc[train['V_14'] == 'unknown', 'V_14'] = int(2)
    train.loc[train['V_26'] == 'unknown', 'V_26'] = int(2)
    train['V_28'] = train['V_28'].fillna('2800')
    train['V_4'] = train['V_4'].fillna('0')

    # 非数值型转为数值型
    train['V_4'] = train['V_4'].astype('float64')
    train['V_8'] = train['V_8'].astype('int')
    train['V_10'] = train['V_10'].astype('int')
    train['V_14'] = train['V_14'].astype('int')
    train['V_26'] = train['V_26'].astype('int')
    train['V_28'] = train['V_28'].astype('int')

    # 缺失值填充
    # 随机森林预测V_4中的缺失值
    from sklearn.ensemble import RandomForestRegressor
    # 构建新的特征矩阵，特征选择原则：与V_4的相关系数>0.008
    v4 = train[['V_4', 'V_5', 'V_6', 'V_8', 'V_22', 'V_23', 'V_29']]
    # 获得v_4列中值不为0的行
    v4_known = v4[~(v4.V_4 == 0)]
    # 获得v_4列中值为0的行
    v4_unknown = v4[v4['V_4'].isin([0])]
    # X为输入的特征属性 Y为目标
    X = v4_known.values[:, 1:]
    Y = v4_known.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, min_samples_split=2)
    # 根据已知数据拟合随机森林模型
    rfr.fit(X, Y)
    # 用得到的模型进行未知部分的预测
    predictV4 = rfr.predict(v4_unknown.values[:, 1:])
    # 用得到的预测结果填补原缺失数据
    train.loc[train['V_4'] == 0, 'V_4'] = predictV4

    # 随机森林预测V_10中的缺失值
    v10 = train[['V_10', 'V_14', 'V_5', 'V_6', 'V_8', 'V_30']]
    v10_known = v10[v10['V_10'].isin([0, 1])]
    v10_unknown = v10[v10['V_10'].isin([2])]
    X = v10_known.values[:,1:]
    Y = v10_known.values[:,0]
    rfr = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, min_samples_split=2)
    rfr.fit(X, Y)
    predictV10 = rfr.predict(v10_unknown.values[:, 1:])
    train.loc[train['V_10'] == 2, 'V_10'] = predictV10

    # 随机森林预测V_14中的缺失值
    v14 = train[['V_14', 'V_10', 'V_5', 'V_6', 'V_8', 'V_15', 'V_30']]
    v14_known = v14[v14['V_14'].isin([0, 1])]
    v14_unknown = v14[v14['V_14'].isin([2])]
    X = v14_known.values[:, 1:]
    Y = v14_known.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, min_samples_split=2)
    rfr.fit(X, Y)
    predictV14 = rfr.predict(v14_unknown.values[:, 1:])
    train.loc[train['V_14'] == 2, 'V_14'] = predictV14

    # 随机森林预测V_26中的缺失值
    v26 = train[['V_26', 'V_5', 'V_6', 'V_8', 'V_19']]
    v26_known = v26[v26['V_26'].isin([0, 1])]
    v26_unknown = v26[v26['V_26'].isin([2])]
    X = v26_known.values[:, 1:]
    Y = v26_known.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, min_samples_split=2)
    rfr.fit(X, Y)
    predictV26 = rfr.predict(v26_unknown.values[:, 1:])
    train.loc[train['V_26'] == 2, 'V_26'] = predictV26

    # 随机森林预测V_28中的缺失值
    v28 = train[['V_28', 'V_5', 'V_6', 'V_8', 'V_15', 'V_16', 'V_24']]
    v28_known = v28[~(v28.V_28 == 2800)]
    v28_unknown = v28[v28['V_28'].isin([2800])]
    X = v28_known.values[:, 1:]
    Y = v28_known.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, min_samples_split=2)
    rfr.fit(X, Y)
    predictV28 = rfr.predict(v28_unknown.values[:, 1:])
    train.loc[train['V_28'] == 2800, 'V_28'] = predictV28

    train_ = {}
    for i in train.columns:
        train_i = train[i]
        if any(train[i].isna()):
            train_i = train_i.replace(np.nan, train[i].mean())
        if len(train_i.value_counts()) <= 20 and train_i.dtype != object:
            train_i = train_i.astype(int)
        train_[i] = train_i
    return pd.DataFrame(train_)
train = build_data(train)
test = build_data(test)

#预处理文件存档
save = pd.DataFrame(train)
save.to_csv('train_input.csv', index=0)
save = pd.DataFrame(test)
save.to_csv('test_input.csv', index=0)

print("数据已生成！")

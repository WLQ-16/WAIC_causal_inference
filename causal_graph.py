# 本实验使用了causal-learn工具辅助构建因果图,具体安装步骤请见:https://causal-learn.readthedocs.io/en/latest/getting_started.html
# 基础包的调用
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

# causallearn包的调用
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci  # 导入不同的独立性测试方法
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

data_path = "dataset/train_input.csv" # 需要先通过数据预处理(data_processing.py)生成处理后的csv文件

# 使用FCI方法
# 使用方法详见https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html
# 由于电脑内存不足,无法直接跑出全部节点的因果关系,这里以6个一组进行循环,共跑出7张因果图,保存在"生成的因果图/FCI"文件夹中
for epochs in range(0, 7):
    i = epochs * 6  # 保证每组有6个因子
    if epochs != 6:  # 前6组正常按照顺序选择
        example_data = np.loadtxt(data_path, delimiter=',', skiprows=1, max_rows=8000,
                                  usecols=[i, i + 1, i + 2, i + 3, i + 4, i + 5, 40, 41])  # 不读取标题，全用10000
        # 行数据进行实验（过多会对结果产生影响） ，通过对usecols的更改,可以查看特定节点与treatment和outcome之间的独立关系
        G, edges = fci(example_data, fisherz, 0.05, verbose=False)  # 先调用fci
        pdy = GraphUtils.to_pydot(G)  # 得到因果图
        nodes = G.get_nodes()  # 获取因果图中的节点
        bk = BackgroundKnowledge().add_forbidden_by_node(nodes[7], nodes[6])  # 设置背景知识（约束）:treatment的方向必须指向outcome
        G_with_background_knowledge, edges = fci(example_data, fisherz, 0.05, verbose=True,
                                                 background_knowledge=bk)  # 将背景知识加入fci方法中,再次调用fci方法
        list_dot = ['V_{}'.format(i), 'V_{}'.format(i + 1), 'V_{}'.format(i + 2), 'V_{}'.format(i + 3),
                    'V_{}'.format(i + 4), 'V_{}'.format(i + 5), 'X', 'Y']  # 设置标签
        pdy = GraphUtils.to_pydot(G_with_background_knowledge, labels=list_dot)  # 设置图中的节点标签,其中treatment用X代替,outcome用Y代替
        pdy.write_png("生成的因果图/FCI/node{}~node{}.png".format(i, i + 5))  # 画出因果图并保存
    else:  # 对于最后一组，计算从V_34到V_39，其余代码与上面一致
        example_data = np.loadtxt(data_path, delimiter=',', skiprows=1, max_rows=8000,
                                  usecols=[i - 2, i - 1, i, i + 1, i + 2, i + 3, 40, 41])
        G, edges = fci(example_data, fisherz, 0.05, verbose=False)
        pdy = GraphUtils.to_pydot(G)
        nodes = G.get_nodes()
        bk = BackgroundKnowledge().add_forbidden_by_node(nodes[7], nodes[6])
        G_with_background_knowledge, edges = fci(example_data, fisherz, 0.05, verbose=True, background_knowledge=bk)
        list_dot = ['V_{}'.format(i - 2), 'V_{}'.format(i - 1), 'V_{}'.format(i), 'V_{}'.format(i + 1),
                    'V_{}'.format(i + 2), 'V_{}'.format(i + 3), 'X', 'Y']
        pdy = GraphUtils.to_pydot(G_with_background_knowledge, labels=list_dot)
        pdy.write_png("生成的因果图/FCI/node{}~node{}.png".format(i - 2, i + 3))

# 使用CD—NOD算法
# 使用方法与原理说明见：https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/CDNOD.html
# 方法原理与fci算法近似一致，因此注释中只解释其独有的部分
for epochs in range(0, 7):
    i = epochs * 6
    if epochs != 6:
        example_data = np.loadtxt(data_path, delimiter=',', skiprows=1, max_rows=10000,
                                  usecols=[i, i+1, i+2, i+3, i+4, i+5, 40, 41])
        c_indx = np.reshape(np.asarray(list(range(example_data.shape[0]))), (example_data.shape[0], 1))
        # 该方法的不同点是捕获未观察到的变化因素的时间索引或域索引，在图中用'other'进行表示
        cg = cdnod(example_data, c_indx, 0.05, fisherz, True, 0, -1)
        nodes = cg.G.get_nodes()
        bk = BackgroundKnowledge().add_forbidden_by_node(nodes[7], nodes[6])
        cg = cdnod(example_data, c_indx, 0.05, fisherz, True, 0, -1, background_knowledge=bk)
        list_dot = ['V_{}'.format(i), 'V_{}'.format(i+1), 'V_{}'.format(i+2), 'V_{}'.format(i+3), 'V_{}'.format(i+4),
                    'V_{}'.format(i+5), 'X', 'Y', 'other']
        pic = GraphUtils.to_pydot(cg.G, labels=list_dot)
        pic.write_png("生成的因果图/CD-NOD/node{}~node{}.png".format(i, i+5))
    else:
        example_data = np.loadtxt(data_path, delimiter=',', skiprows=1, max_rows=10000,
                                  usecols=[i - 2, i - 1, i, i + 1, i + 2, i + 3, 40, 41])
        c_indx = np.reshape(np.asarray(list(range(example_data.shape[0]))), (example_data.shape[0], 1))
        cg = cdnod(example_data, c_indx, 0.05, fisherz, True, 0, -1)
        nodes = cg.G.get_nodes()
        bk = BackgroundKnowledge().add_forbidden_by_node(nodes[7], nodes[6])
        cg = cdnod(example_data, c_indx, 0.05, fisherz, True, 0, -1, background_knowledge=bk)
        list_dot = ['V_{}'.format(i - 2), 'V_{}'.format(i - 1), 'V_{}'.format(i), 'V_{}'.format(i + 1),
                    'V_{}'.format(i + 2), 'V_{}'.format(i + 3), 'X', 'Y', 'other']
        pic = GraphUtils.to_pydot(cg.G, labels=list_dot)
        pic.write_png("生成的因果图/CD-NOD/node{}~node{}.png".format(i-2, i + 3))

# 使用PC方法
# 算法介绍与使用方法见：https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html
# PC算法使用的库在循环画图时会累计之前的因果关系，导致后面的结果不准，因此本代码以第一组为例进行演示
example_data = np.loadtxt(data_path, delimiter=',', skiprows=1, max_rows=10000,usecols=(0, 1, 2, 3, 4, 5, 40, 41))
# 导入数据，取10000行进行分析
example_result = pc(example_data, 0.05, fisherz, True, 0, -1)
# 初步调用pc
nodes = example_result.G.get_nodes()
# 获取节点
bk = BackgroundKnowledge().add_forbidden_by_node(nodes[7], nodes[6])
# 获取背景知识
example_result = pc(example_data, 0.05, fisherz, True, 0, -1, background_knowledge=bk)
# 将背景知识加入pc算法中
example_result.to_nx_graph()
example_result.draw_nx_graph(skel=False)
plt.savefig("生成的因果图/PC/node{}~node{}.png".format(0, 5))
# 画出因果图

# 以下两种方法无法将完全一致的列加入计算，因此先从数据集中去除V_20,V_26和V_27
example_data = pd.read_csv('dataset/train_input.csv', nrows=10000, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                                  17, 18, 19, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41))
list_dot = ['V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9', 'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18',
    'V_19', 'V_21', 'V_22', 'V_23', 'V_24', 'V_25', 'V_28', 'V_29', 'V_30', 'V_31', 'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39', 'x', 'y']
#
# 使用GRaSP方法
# 方法的原理和介绍见：https://causal-learn.readthedocs.io/en/latest/search_methods_index/Permutation-based%20causal%20discovery%20methods/GRaSP.html#algorithm-introduction
G = grasp(example_data, 'local_score_BIC')
# 调用GRaSP方法对数据集进行分析，得分函数使用local_score_BIC
pyd = GraphUtils.to_pydot(G, labels=list_dot)
# 用因果图的形式呈现，加上各元素的标签
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
# 将生成的GRaS因果图保存在文件夹中
plt.savefig("生成的因果图/GRaS/GRaS_cg.png")
plt.show()
#
# # 使用GES方法
# # 方法和原理介绍见：https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/GES.html
Record = ges(example_data, score_func='local_score_BIC', maxP=None, parameters='kfold')
# 调用GES方法对数据集进行分析，得分函数使用local_score_BIC
pyd = GraphUtils.to_pydot(Record['G'], labels=list_dot)
# 用因果图的形式呈现，加上各元素的标签
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
# 将生成的GES因果图保存在文件夹中
plt.savefig("生成的因果图/GRaS/GRaS_cg.png")
plt.imshow(img)
plt.show()

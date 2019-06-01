### <center>K-近邻算法</center>
> ##### 选取一个最近的点成为最近邻算法，k个点才叫k近邻算法！

#### 一、概述
##### K-近邻算法的步骤：
> 1. 计算已知类别数据集中的点与当前点之间的距离
> 2. 按照距离递增次序排序
> 3. 选取与当前距离最小的k个点
> 4. 确定前k个点所在类别的出现频率
> 5. 返回前k个点出现频率最高的类别作为当前点的预测类别

#### 二、python实现
```
python实现
# 1.构建数据集
pd
import pandas as pd
rowdata={'电影名称':['无问西东','后来的我们','前任3','红海行动','唐人街探案','战狼2'],
        '打斗镜头':[1,5,12,108,112,115],
        '接吻镜头':[101,89,97,5,9,8],
        '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}
movie_data = pd.DataFrame(rowdata)
movie_data
movie_data
电影名称	打斗镜头	接吻镜头	电影类型
0	无问西东	1	101	爱情片
1	后来的我们	5	89	爱情片
2	前任3	12	97	爱情片
3	红海行动	108	5	动作片
4	唐人街探案	112	9	动作片
5	战狼2	115	8	动作片
2.
# 2.计算已知类别数据集中的点与当前点之间的距离
new_data = [24, 67]
dist = list((((movie_data.iloc[:6, 1:3] - new_data)**2).sum(1))**0.5)
dist
dist
[41.048751503547585,
 29.068883707497267,
 32.31098884280702,
 104.4030650891055,
 105.39449701004318,
 108.45275469069469]
K个点
# 3.将距离升序排列，然后选取距离最小的K个点
dist_l = pd.DataFrame({'dist': dist, 'labels': (movie_data.iloc[:6, 3])})
dr = dist_l.sort_values(by = 'dist')[:4]
dr
dist	labels
1	29.068884	爱情片
2	32.310989	爱情片
0	41.048752	爱情片
3	104.403065	动作片
的出现频率
# 4.确定前K个点所在类别的出现频率
re = dr.loc[:, 'labels'].value_counts()
re
re
爱情片    3
动作片    1
Name: labels, dtype: int64
作为当前点的预测类别
# 5.选取频率最高的类别作为当前点的预测类别
result
result = []
result.append(re.index[0])
result
['爱情片']
```
#### 三、函数封装
```
K-近邻算法函数封装
import pandas as pd
# ```
# 函数功能：KNN分类器
# 参数说明：
#     new_data：需要预测分类的数据集
#     dataSet：已知分类标签的数据集（训练集）
#     k：K-近邻算法参数，选择距离最小的k个点
# 返回：
#     result：分类结果
# ```
def classify(inX, dataSet, k):
    result = []
    dist = list((((dataSet.iloc[:, 1:3] - inX)**2).sum(1))**0.5)
    dist_l = pd.DataFrame({'dist': dist, 'labels': (dataSet.iloc[:, 3])})
    dr = dist_l.sort_values(by = 'dist')[:k]
    re = dr.loc[:,'labels'].value_counts()
    result.append(re.index[0])
    return result
rowdata={'电影名称':['无问西东','后来的我们','前任3','红海行动','唐人街探案','战狼2'],
        '打斗镜头':[1,5,12,108,112,115],
        '接吻镜头':[101,89,97,5,9,8],
        '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}
movie_data = pd.DataFrame(rowdata)
new_data = [24, 67]
classify(new_data, movie_data, 4)
rowdata={'电影名称':['无问西东','后来的我们','前任3','红海行动','唐人街探案','战狼2'],
        '打斗镜头':[1,5,12,108,112,115],
        '接吻镜头':[101,89,97,5,9,8],
        '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}
movie_data = pd.DataFrame(rowdata)
new_data = [24, 67]
classify(new_data, movie_data, 4)
```
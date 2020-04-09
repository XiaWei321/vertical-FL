import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../企业数据特征记录.csv')
df.drop(df.columns[17:138], axis=1, inplace=True)  # 删除乱码
del df['公司名称']
del df['数据的时间']

Y = df['信用评级']
grade = ['C', 'CC', 'CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']
for i in range(len(df['信用评级'])):
    Y[i] = grade.index(Y[i])
del df['信用评级']

# 所有数据原来是string类型的，转换成数值类型
for index, row in df.iteritems():
    df[index] = pd.to_numeric(df[index])

# 将所有控制部分用平均值替补
for column in list(df.columns[df.isnull().sum() > 0]):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)

# 营业利润率
outliers = df[df['营业利润率（%）'] < df['营业利润率（%）'].quantile(0.05)].index
df.loc[outliers, '营业利润率（%）'] = df['营业利润率（%）'].quantile(0.05)
scaler = MinMaxScaler()
df['营业利润率（%）'] = scaler.fit_transform(df['营业利润率（%）'].values.reshape(-1, 1))

# 净资产收益率
outliers = df[df['净资产收益率（%）'] < df['净资产收益率（%）'].quantile(0.10)].index
df.loc[outliers, '净资产收益率（%）'] = df['净资产收益率（%）'].quantile(0.10)
scaler = MinMaxScaler()
df['净资产收益率（%）'] = scaler.fit_transform(df['净资产收益率（%）'].values.reshape(-1, 1))

# 资产报酬率
outliers = df[df['资产报酬率（%）'] > df['资产报酬率（%）'].quantile(0.95)].index
df.loc[outliers, '资产报酬率（%）'] = df['资产报酬率（%）'].quantile(0.95)
scaler = MinMaxScaler()
df['资产报酬率（%）'] = scaler.fit_transform(df['资产报酬率（%）'].values.reshape(-1, 1))

# 速动比率
outliers = df[df['速动比率（%）'] > df['速动比率（%）'].quantile(0.95)].index
df.loc[outliers, '速动比率（%）'] = df['速动比率（%）'].quantile(0.95)
scaler = MinMaxScaler()
df['速动比率（%）'] = scaler.fit_transform(df['速动比率（%）'].values.reshape(-1, 1))

# 现金比率
outliers = df[df['现金比率（%）'] > df['现金比率（%）'].quantile(0.91)].index
df.loc[outliers, '现金比率（%）'] = df['现金比率（%）'].quantile(0.91)
scaler = MinMaxScaler()
df['现金比率（%）'] = scaler.fit_transform(df['现金比率（%）'].values.reshape(-1, 1))

# 资产负债率
outliers = df[df['资产负债率（%）'] > df['资产负债率（%）'].quantile(0.98)].index
df.loc[outliers, '资产负债率（%）'] = df['资产负债率（%）'].quantile(0.98)
scaler = MinMaxScaler()
df['资产负债率（%）'] = scaler.fit_transform(df['资产负债率（%）'].values.reshape(-1, 1))

# 产权比率
outliers = df[df['产权比率（%）'] < df['产权比率（%）'].quantile(0.02)].index
df.loc[outliers, '产权比率（%）'] = df['产权比率（%）'].quantile(0.02)
outliers = df[df['产权比率（%）'] > df['产权比率（%）'].quantile(0.92)].index
df.loc[outliers, '产权比率（%）'] = df['产权比率（%）'].quantile(0.92)
scaler = MinMaxScaler()
df['产权比率（%）'] = scaler.fit_transform(df['产权比率（%）'].values.reshape(-1, 1))

# 净利润增长率
outliers = df[df['净利润增长率（%）'] > df['净利润增长率（%）'].quantile(0.98)].index
df.loc[outliers, '净利润增长率（%）'] = df['净利润增长率（%）'].quantile(0.98)
outliers = df[df['净利润增长率（%）'] < df['净利润增长率（%）'].quantile(0.08)].index
df.loc[outliers, '净利润增长率（%）'] = df['净利润增长率（%）'].quantile(0.08)
scaler = MinMaxScaler()
df['净利润增长率（%）'] = scaler.fit_transform(df['净利润增长率（%）'].values.reshape(-1, 1))

# 总资产增长率
outliers = df[df['总资产增长率（%）'] > df['总资产增长率（%）'].quantile(0.97)].index
df.loc[outliers, '总资产增长率（%）'] = df['总资产增长率（%）'].quantile(0.97)
outliers = df[df['总资产增长率（%）'] < df['总资产增长率（%）'].quantile(0.01)].index
df.loc[outliers, '总资产增长率（%）'] = df['总资产增长率（%）'].quantile(0.01)
scaler = MinMaxScaler()
df['总资产增长率（%）'] = scaler.fit_transform(df['总资产增长率（%）'].values.reshape(-1, 1))

# 应收账款周转率
outliers = df[df['应收账款周转率（次）'] > df['应收账款周转率（次）'].quantile(0.90)].index
df.loc[outliers, '应收账款周转率（次）'] = df['应收账款周转率（次）'].quantile(0.90)
scaler = MinMaxScaler()
df['应收账款周转率（次）'] = scaler.fit_transform(df['应收账款周转率（次）'].values.reshape(-1, 1))

# 总资产周转率
outliers = df[df['总资产周转率（次）'] > df['总资产周转率（次）'].quantile(0.90)].index
df.loc[outliers, '总资产周转率（次）'] = df['总资产周转率（次）'].quantile(0.90)
scaler = MinMaxScaler()
df['总资产周转率（次）'] = scaler.fit_transform(df['总资产周转率（次）'].values.reshape(-1, 1))

# 流动资产周转率
outliers = df[df['流动资产周转率（次）'] > df['流动资产周转率（次）'].quantile(0.94)].index
df.loc[outliers, '流动资产周转率（次）'] = df['流动资产周转率（次）'].quantile(0.94)
scaler = MinMaxScaler()
df['流动资产周转率（次）'] = scaler.fit_transform(df['流动资产周转率（次）'].values.reshape(-1, 1))

# 前五大股东持股总和
scaler = MinMaxScaler()
df['前五大股东持股总和占比（%）'] = scaler.fit_transform(df['前五大股东持股总和占比（%）'].values.reshape(-1, 1))

print(df['前五大股东持股总和占比（%）'])
# 国有控股占比
del df['国有控股占比（%）']


print(df.info())
# 数据集x

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.2, random_state=100)
X_train.to_csv('../data/train_data.csv', header=True, index=False)
Y_train.to_csv('../data/train_label.csv', header=True, index=False)
X_test.to_csv('../data/test_data.csv', header=True, index=False)
Y_test.to_csv('../data/test_label.csv', header=True, index=False)


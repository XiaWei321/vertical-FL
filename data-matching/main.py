import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('ggplot')

df = pd.read_csv('./input/loan_final313.csv')
df.drop('id', axis=1, inplace=True)
df.drop('year', axis=1, inplace=True)
df.drop('issue_d', axis=1, inplace=True)
df.drop('final_d', axis=1, inplace=True)

scaler = MinMaxScaler()
df['emp_length_int'] = scaler.fit_transform(df['emp_length_int'].values.reshape(-1, 1))  # 缺少script包

df = pd.concat([df, pd.get_dummies(df['home_ownership'])], axis=1).drop(['home_ownership', 'home_ownership_cat'],
                                                                        axis=1)
df.drop(['OTHER', 'NONE', 'ANY'], axis=1, inplace=True)

#  income_category
df = pd.concat([df, pd.get_dummies(df['income_category'])], axis=1).drop(['income_category', 'income_cat'], axis=1)

#  annual_inc
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['annual_inc'])
outliers = df[df['annual_inc'] > df['annual_inc'].quantile(0.99)].index
df.loc[outliers, 'annual_inc'] = df['annual_inc'].quantile(0.99)
scaler = MinMaxScaler()
df['annual_inc'] = scaler.fit_transform(df['annual_inc'].values.reshape(-1, 1))

plt.figure(figsize=(12, 6))
sns.boxplot(x=df['annual_inc'])

#  loan_amount
scaler = MinMaxScaler()
df['loan_amount'] = scaler.fit_transform(df['loan_amount'].values.reshape(-1, 1))

df = pd.concat([df, pd.get_dummies(df['term_cat'], prefix='term')], axis=1).drop(['term', 'term_cat'], axis=1)
df.drop(['application_type', 'application_type_cat'], axis=1, inplace=True)

#  purpose
df = pd.concat([df, pd.get_dummies(df['purpose'])], axis=1).drop(['purpose', 'purpose_cat'], axis=1)
df.drop(['car', 'small_business', 'other', 'wedding', 'home_improvement', 'major_purchase',
         'medical', 'moving', 'vacation', 'house', 'renewable_energy',
         'educational'], axis=1, inplace=True)

#  interest_payments
df = pd.concat([df, pd.get_dummies(df['interest_payments'], prefix='int')], axis=1).drop(
    ['interest_payments', 'interest_payment_cat'], axis=1)
df.drop('int_High', axis=1, inplace=True)

df.drop('loan_condition', axis=1, inplace=True)

#  interest_rate
outliers = df[df['interest_rate'] > df['interest_rate'].quantile(.99)].index
df.loc[outliers, 'interest_rate'] = df['interest_rate'].quantile(.99)
scaler = MinMaxScaler()
df['interest_rate'] = scaler.fit_transform(df['interest_rate'].values.reshape(-1, 1))

df.drop('grade', axis=1, inplace=True)
df.drop('grade_cat', axis=1, inplace=True)

# dti
outliers = df[df['dti'] > df['dti'].quantile(.99)].index
df.loc[outliers, 'dti'] = df['dti'].quantile(.99)
scaler = MinMaxScaler()
df['dti'] = scaler.fit_transform(df['dti'].values.reshape(-1, 1))

df.drop('total_pymnt', axis=1, inplace=True)
df.drop('total_rec_prncp', axis=1, inplace=True)
df.drop('recoveries', axis=1, inplace=True)
df.drop('installment', axis=1, inplace=True)
df.drop('region', axis=1, inplace=True)

# 数据集

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

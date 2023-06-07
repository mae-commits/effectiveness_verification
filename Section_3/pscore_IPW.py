import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from tqdm import tqdm_notebook
import plotly.express as px
import plotly.graph_objects as go
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
# セレクションバイアスのあるデータの作成
mail_df = pd.read_csv('http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv')
### 女性向けメールが配信されたデータを削除したデータを作成
male_df = mail_df[mail_df.segment != 'Womens E-Mail'].copy() # 女性向けメールが配信されたデータを削除
male_df['treatment'] = male_df.segment.apply(lambda x: 1 if x == 'Mens E-Mail' else 0) #介入を表すtreatment変数を追加
## バイアスのあるデータの作成
sample_rules = (male_df.history > 300) | (male_df.recency < 6) | (male_df.channel=='Multichannel')
biased_df = pd.concat([
    male_df[(sample_rules) & (male_df.treatment == 0)].sample(frac=0.5, random_state=1),
    male_df[(sample_rules) & (male_df.treatment == 1)],
    male_df[(~sample_rules) & (male_df.treatment == 0)],
    male_df[(~sample_rules) & (male_df.treatment == 1)].sample(frac=0.5, random_state=1)
], axis=0, ignore_index=True)

def get_ipw(X, y, random_state=0):
    # 傾向スコアを計算
    ps_model = LogisticRegression(solver='lbfgs', random_state=random_state).fit(X, y)
    ps_score = ps_model.predict_proba(X)[:, 1]
    all_df = pd.DataFrame({'treatment': y, 'ps_score': ps_score})
    treatments =all_df.treatment.unique() # 介入が 0 or 1 の2値になっているかどうかを判断
    if len(treatments) != 2:
        print('2群のマッチングしかできません。2群は必ず[0, 1]で表現してください。')
        raise ValueError
    # group 1 for treatment == 1, group 2 for treatment == 0
    group_1_df = all_df[all_df.treatment == 1].copy()
    group_2_df = all_df[all_df.treatment == 0].copy()
    # IPW を計算
    group_1_df['weight'] = 1 / group_1_df.ps_score
    group_2_df['weight'] = 1 / group_2_df.ps_score
    # group 1, group 2 の weight の値を結合
    weights = pd.concat([group_1_df, group_2_df]).sort_index()['weight'].values
    return weights

y = biased_df['treatment']
X = pd.get_dummies(biased_df[['recency', 'channel', 'history']], columns=['channel'], drop_first=True, dtype=float)
weights = get_ipw(X, y)

# 重み付きデータでの効果の推定
y = biased_df.spend
X = biased_df.treatment
X = sm.add_constant(X)
results = sm.WLS(y, X, weights=weights).fit()
coef = results.summary().tables[1]
print(coef)
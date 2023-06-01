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

# 傾向スコアの推定
y = biased_df['treatment']
X = pd.get_dummies(biased_df[['recency', 'channel', 'history']], columns=['channel'], drop_first=True, dtype=float)

# 傾向スコアマッチング

def get_matched_dfs_using_propensity_score(X,y, random_state=0):
    # 傾向スコアを計算
    ps_model = LogisticRegression(solver='lbfgs', random_state=random_state).fit(X, y)
    ps_score = ps_model.predict_proba(X)[:, 1]
    all_df = pd.DataFrame({'treatment':y, 'ps_score':ps_score})
    treatments = all_df.treatment.unique()
    if len(treatments) !=  2:
        print('2群のマッチングしかできません。2群は必ず[0, 1]で表現してください。' )
        raise ValueError
    # treatment == 1をgroup1, treatment == 0をgroup2とする。group1にマッチするgroup2を抽出するのでATTの推定になるはず
    group1_df = all_df[all_df.treatment==1].copy()
    group1_indices = group1_df.index
    group1_df = group1_df.reset_index(drop=True)
    group2_df = all_df[all_df.treatment==0].copy()
    group2_indices = group2_df.index
    group2_df = group2_df.reset_index(drop=True)
    

    # 全体の傾向スコアの標準偏差 * 0.2をしきい値とする
    threshold = all_df.ps_score.std() * 0.2

    matched_group1_dfs = []
    matched_group2_dfs = []
    _group1_df = group1_df.copy()
    _group2_df = group2_df.copy()
    
    
    while True:
        # NearestNeighborsで最近傍点1点を見つけ、マッチングする
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(_group1_df.ps_score.values.reshape(-1, 1))
        distances, indices = neigh.kneighbors(_group2_df.ps_score.values.reshape(-1, 1))
        # 重複点を削除する
        distance_df = pd.DataFrame({'distance': distances.reshape(-1), 'indices': indices.reshape(-1)})
        distance_df.index = _group2_df.index
        distance_df = distance_df.drop_duplicates(subset='indices')
        # しきい値を超えたレコードを削除する
        distance_df = distance_df[distance_df.distance < threshold]
        if len(distance_df) == 0:
            break
        # マッチングしたレコードを抽出、削除する
        group1_matched_indices = _group1_df.iloc[distance_df['indices']].index.tolist()
        group2_matched_indices = distance_df.index
        matched_group1_dfs.append(_group1_df.loc[group1_matched_indices])
        matched_group2_dfs.append(_group2_df.loc[group2_matched_indices])
        _group1_df = _group1_df.drop(group1_matched_indices)
        _group2_df = _group2_df.drop(group2_matched_indices)

    # マッチしたレコードを返す
    group1_df.index = group1_indices
    group2_df.index = group2_indices
    matched_df = pd.concat([
        group1_df.iloc[pd.concat(matched_group1_dfs).index],
        group2_df.iloc[pd.concat(matched_group2_dfs).index]
    ]).sort_index()
    matched_indices = matched_df.index

    return X.loc[matched_indices], y.loc[matched_indices]

matchX, matchy  = get_matched_dfs_using_propensity_score(X, y)

# マッチング後のデータで効果の推定
y = biased_df.loc[matchX.index].spend
X = matchy
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
coef = results.summary().tables[1]

print(coef)
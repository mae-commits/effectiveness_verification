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

# 共変量のバランスを確認

def calc_absolute_mean_difference(df):
    # (treatment 群の平均 - control 群の平均) /全体の標準偏差
    return ((df[df.treatment==1].drop('treatment', axis=1).mean() - (df[df.treatment==0].drop('treatment', axis=1).mean())) \
        / df.drop('treatment', axis=1).std()).abs()

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

unadjusted_df = pd.get_dummies(biased_df[['treatment', 'recency', 'channel', 'history']], columns=['channel'],dtype=float)
unadjusted_amd = calc_absolute_mean_difference(unadjusted_df)

matchX, matchy  = get_matched_dfs_using_propensity_score(X, y)

# マッチング後のデータで効果の推定
y = biased_df.loc[matchX.index].spend
X = matchy
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
coef = results.summary().tables[1]

# 傾向スコアマッチング後の Absolute Mean Difference
after_matching_df = pd.get_dummies(biased_df.loc[matchX.index][['treatment', 'recency', 'history', 'channel']], columns=['channel'], dtype=float)
after_matching_amd = calc_absolute_mean_difference(after_matching_df)

# IPW で重み付け後の Absolute Mean Difference
# 重みのぶんレコードを増やして計算する
after_weighted_df = pd.get_dummies(biased_df[['treatment', 'recency', 'channel', 'history']], columns=['channel'])
weights_int = (weights * 100).astype(int)
weighted_df = []
for i, value in enumerate(after_weighted_df.values):
    weighted_df.append(np.tile(value, (weights_int[i], 1)))
weighted_df = np.concatenate(weighted_df).reshape(-1, 6)
weighted_df = pd.DataFrame(weighted_df)
weighted_df.columns = after_weighted_df.columns
after_weighted_amd = calc_absolute_mean_difference(weighted_df)

balance_df = pd.concat([
    pd.DataFrame({'Absolute Mean Difference': unadjusted_amd, 'Sample': 'Unadjusted'}),
    pd.DataFrame({'Absolute Mean Difference': after_matching_amd, 'Sample': 'Adjusted'})
])
fig = px.scatter(balance_df, x='Absolute Mean Difference', y=balance_df.index, color='Sample',
                title='3.5 マッチングしたデータでの共変量のバランス')
fig.show()
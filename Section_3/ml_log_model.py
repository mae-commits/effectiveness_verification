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

random_state = 0

# 学習エータと配信ログを作るデータに分割
male_df_train, male_df_test = train_test_split(male_df, test_size=0.5, random_state=random_state)
male_df_train = male_df_train[male_df_train.treatment == 0]

# 売上が発生する確率を予測するモデルを作成
model = LogisticRegression(random_state=random_state)
y_train = male_df_train['conversion']
X_train = pd.get_dummies(
    male_df_train[['recency', 'history_segment', 'channel', 'zip_code']], columns=['history_segment', 'channel', 'zip_code'], drop_first=True, dtype=float
)

X_test = pd.get_dummies(
    male_df_test[['recency', 'history_segment', 'channel', 'zip_code']], columns=['history_segment', 'channel', 'zip_code'], drop_first=True, dtype=float
)

model.fit(X_train, y_train)

# 売上の発生確率からメールの配信確率を決める
pred_cv = model.predict_proba(X_test)[:, 1] # ロジスティック回帰を元に各サンプルの配信確率を決定
pred_cv_rank = pd.Series(pred_cv, name='proba').rank(pct=True) # 各サンプルの配信確率をランク付けし、予測値とする
# 配信確率を元にメールの配信を決める
mail_assign = pred_cv_rank.apply(lambda x: np.random.binomial(n=1, p=x))

# 配信ログを作成
male_df_test['mail_assign'] = mail_assign
male_df_test['ps'] = pred_cv_rank

ml_male_df = male_df_test[
    ((male_df_test.treatment == 1) & (male_df_test.mail_assign == 1)) |
    ((male_df_test.treatment == 0) & (male_df_test.mail_assign == 0))
].copy()

# 平均の比較
# 実験をしていた場合の平均の差を確認
y = male_df_test.spend
X = male_df_test.treatment
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
coef = results.summary().tables[1]
print("RCT model")
print(coef)

## セレクションバイアスの影響を受けている平均の比較
y_new = ml_male_df.spend
X_new = ml_male_df.treatment
X_new = sm.add_constant(X_new)
results = sm.OLS(y_new, X_new).fit()
coef_new = results.summary().tables[1]
print("ml model (biased)")
print(coef_new)

# 傾向スコアマッチングの推定（TPS）
def get_matched_dfs_using_obtained_propensity_score(X, y, ps_score, random_state=0):
    all_df = pd.DataFrame({'treatment': y, 'ps_score': ps_score})
    treatments = all_df.treatment.unique()
    if len(treatments) != 2:
        print('2群のマッチングしかできません。2群は必ず[0, 1]で表現してください。')
        raise ValueError
    # treatment == 1 を group1, treatment == 0 を group2 とする。group1 にマッチする group2 を抽出するのでATTの推定になるはず
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

matchX, matchy = get_matched_dfs_using_obtained_propensity_score(ml_male_df, ml_male_df.treatment, ps_score=ml_male_df.ps)

## マッチング後のデータで効果の推定
y_match = matchX.spend
X_match = matchy
X_match = sm.add_constant(X_match)
results = sm.OLS(y_match, X_match).fit()
coef_match = results.summary().tables[1]

print("matching")
print(coef_match)

# IPW の推定

def get_ipw_obtained_ps(X, y, ps_score, random_state=0):
    all_df = pd.DataFrame({'treatment': y, 'ps_score': ps_score})
    treatments = all_df.treatment.unique()
    if len(treatments) != 2:
        print('2群のマッチングしかできません。2群は必ず[0, 1]で表現してください。')
        raise ValueError
    # treatment == 1をgroup1, treatment == 0をgroup2とする。
    group1_df = all_df[all_df.treatment==1].copy()
    group2_df = all_df[all_df.treatment==0].copy()
    group1_df['weight'] = 1 / group1_df.ps_score
    group2_df['weight'] = 1 / (1 - group2_df.ps_score)
    weights = pd.concat([group1_df, group2_df]).sort_index()['weight'].values
    return weights

weights = get_ipw_obtained_ps(ml_male_df, ml_male_df.treatment, ps_score=ml_male_df.ps)
## 重み付きデータでの効果の推定
y_ipw = ml_male_df.spend
X_ipw = ml_male_df.treatment
X_ipw = sm.add_constant(X_ipw)
results = sm.WLS(y_ipw, X_ipw, weights=weights).fit()
coef_ipw = results.summary().tables[1]

print("IPW")
print(coef_ipw)
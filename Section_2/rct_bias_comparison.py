import pandas as pd
import statsmodels.api as sm
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

dumped_male_df_path = '../data/male_df.joblib'
dumped_biased_df_path = '../data/biased_df.joblib'

# データセットが規定のディレクトリに存在する場合
if os.path.exists(dumped_male_df_path):
    male_df = joblib.load(dumped_male_df_path)
    biased_df = joblib.load(dumped_biased_df_path)
# 存在しない場合
else:
    # セレクションバイアスのあるデータの作成
    mail_df = pd.read_csv('http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv')
    # 女性向けメールが配信されたデータを削除したデータを作成
    male_df = mail_df[mail_df.segment != 'Womens E-Mail'].copy()
    # 介入を表すtreatment変数を追加
    male_df['treatment'] = male_df.segment.apply(lambda x: 1 if x == 'Mens E-Mail' else 0)
    # バイアスのあるデータの作成
    sample_rules = (male_df.history > 300) | (male_df.recency < 6) | (male_df.channel=='Multichannel')
    # 個々の条件を満たすデータの連結
    biased_df = pd.concat([
        male_df[(sample_rules) & (male_df.treatment == 0)].sample(frac=0.5, random_state=1),
        male_df[(sample_rules) & (male_df.treatment == 1)],
        male_df[(~sample_rules) & (male_df.treatment == 0)],
        male_df[(~sample_rules) & (male_df.treatment == 1)].sample(frac=0.5, random_state=1)
    ], axis=0, ignore_index=True)

# RCTデータでの単回帰
y = male_df.spend
X = male_df[['treatment']]
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
rct_reg_coef = results.summary().tables[1]

# バイアスのあるデータでの単回帰
y = biased_df.spend
nonrct_X = biased_df[['treatment']]
nonrct_X = sm.add_constant(nonrct_X)
nonrct_results = sm.OLS(y, nonrct_X).fit()
nonrct_reg_coef = nonrct_results.summary().tables[1]

## バイアスのあるデータでの重回帰
y = biased_df.spend
# R lmではカテゴリ変数は自動的にダミー変数化されているのでそれを再現
X = pd.get_dummies(biased_df[['treatment', 'recency', 'channel', 'history']], columns=['channel'], drop_first=True)
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
nonrct_mreg_coef = results.summary().tables[1]

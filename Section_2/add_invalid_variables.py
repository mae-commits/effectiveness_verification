import pandas as pd
import statsmodels.api as sm
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

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

# visit と treatment の相関
y = biased_df.treatment
X = pd.get_dummies(biased_df[['visit', 'channel', 'recency', 'history']], columns=['channel'], drop_first=True, dtype=float)
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
print(results.summary().tables[1])

# visitを入れた回帰分析を実行
y = biased_df.spend
X = pd.get_dummies(biased_df[['treatment', 'channel', 'recency', 'history', 'visit']], columns=['channel'], drop_first=True, dtype=float)
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
print(results.summary().tables[1])
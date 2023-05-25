import pandas as pd
import statsmodels.api as sm
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

dumped_male_df_path = '../data/male_df.joblib'
dumped_biased_df_path = '../data/biased_df.joblib'

if not os.path.exists(dumped_male_df_path):
    male_df = joblib.load(dumped_male_df_path)
    biased_df = joblib.load(dumped_biased_df_path)
else:
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

# history を抜いた回帰分析とパラメータの取り出し
y = biased_df.spend
# 指定したカテゴリ変数をダミー変数に変更（変更時に型指定する必要あり）
X = pd.get_dummies(biased_df[['treatment', 'recency', 'channel']], columns=['channel'], drop_first=True, dtype=float)
X = sm.add_constant(X)
results =sm.OLS(y, X).fit()
short_coef = results.summary().tables[1]
short_coef_df = pd.read_html(short_coef.as_html(), header=0, index_col=0)[0]

# 介入結果に関するパラメータのみを取り出す
alpha_1 = results.params['treatment']

# history を含む回帰分析とパラメータの取り出し
y = biased_df.spend
# 指定したカテゴリ変数をダミー変数に変更（変更時に型指定する必要あり）
X = pd.get_dummies(biased_df[['treatment', 'recency', 'channel', 'history']], columns=['channel'], drop_first=True, dtype=float)
X = sm.add_constant(X)
results =sm.OLS(y, X).fit()
long_coef = results.summary().tables[1]
long_coef_df = pd.read_html(long_coef.as_html(), header=0, index_col=0)[0]

# 介入結果とhistoryに関するパラメータを取り出す
beta_1 = results.params['treatment']
beta_2 = results.params['history']

# 脱落変数（今回は、history）と介入変数での回帰分析
y = biased_df.history
# 指定したカテゴリ変数をダミー変数に変更（変更時に型指定する必要あり）
X = pd.get_dummies(biased_df[['treatment', 'recency', 'channel']], columns=['channel'], drop_first=True, dtype=float)
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
omitted_coef = results.summary().tables[1]
omitted_coef_df = pd.read_html(omitted_coef.as_html(), header=0, index_col=0)[0]
gamma_1 = results.params['treatment']

# OVB の確認
print(beta_2 * gamma_1)
print(alpha_1 - beta_1)
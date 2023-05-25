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
    
# 回帰分析の実行
y = biased_df.spend # バイアスのあるデータ中での 'spend' 列のデータを被説明変数（目的変数）として抽出
X = biased_df[['treatment','history']] # 共変量Xを定義
X = sm.add_constant(X) # 定数項の追加
model = sm.OLS(y, X) # 最小2乗法による回帰分析
results = model.fit()

# 分析結果のレポート（複数テーブル）
summary = results.summary()
# 推定されたパラメータの取り出し（目的の変数が含まれるテーブルの取り出し）
biased_reg_coef = summary.tables[1]
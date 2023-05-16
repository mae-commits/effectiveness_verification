import pandas as pd
from scipy import stats 
import joblib

# データの読み込み
mail_df = pd.read_csv('http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv')


# (4) データの準備
# 女性向けメールが配信されたデータを削除したデータを作成
male_df = mail_df[mail_df.segment != 'Womens E-Mail'].copy()
# 介入を表すtreatment変数を追加
male_df['treatment'] = male_df.segment.apply(lambda x: 1 if x == 'Mens E-Mail' else 0)

# (5) 集計による比較
# group_by を使ってデータのグループ化
# aggを使って、データの集計
# 引数として設定したものが列の値となる
# treatmrnt は0:男性向けE-Mail が届いた、1：届かなかった
# のグループ分け
male_df.groupby('treatment').agg(
    # グループごとのconversionの平均
    conversion_rate=('conversion', 'mean'),
    # グループごとのspendの平均
    spend_mean=('spend', 'mean'),
    # グループごとのデータ数
    count=('treatment', 'count')
)

# t検定を行う
# (a) 男性向けメールが配信されたグループの購買データを得る
mens_mail = male_df[male_df.treatment == 1].spend.values
# (b) 男性向けメールが配信されなかったグループの購買データを得る
no_mail = male_df[male_df.treatment == 0].spend.values

# (a),(b)の平均の差に対して、有意差検定を実行
stats.ttest_ind(mens_mail, no_mail)

# セレクションバイアスのあるデータの作成
# バイアスのあるデータの作成
# メールを受け取った人の購買金額が大きくなるようにデータの数にバイアスをかけている
# random_state で乱数を発生し、任意のデータを取り出す
sample_rules = (male_df.history > 300) | (male_df.recency < 6) | (male_df.channel=='Miltichannel')
biased_df = pd.concat([
    male_df[(sample_rules) & (male_df.treatment == 0)].sample(frac=0.5, random_state=1),
    male_df[(sample_rules) & (male_df.treatment == 1)],
    male_df[(~sample_rules) & (male_df.treatment == 0)],
    male_df[(~sample_rules) & (male_df.treatment == 1)].sample(frac=0.5, random_state=1)
], axis=0, ignore_index=True)

# セレクションバイアスのあるデータで平均を比較
# groupby を使って集計(Biased)
biased_df.groupby('treatment').agg(
    conversion_rate=('conversion','mean'),
    spend_mean=('spend', 'mean'),
    count=('treatment', 'count')
)

# scipy.stats のttest_ind を使ってt検定の実行

# (a) 男性向けメールが配信されたグループの購買データを得る
mens_mail_biased = biased_df[biased_df.treatment == 1].spend.values
# (b) 男性向けメールが配信されなかったグループの購買データを得る
no_mail_biased = biased_df[biased_df.treatment == 0].spend.values

# (a),(b)の平均の差に対して、有意差検定を実行
stats.ttest_ind(mens_mail_biased, no_mail_biased)

# Section 2 で利用する male_df, biased_df を保存
joblib.dump(male_df, '../data/male_df.joblib')
joblib.dump(biased_df, '../data/biased_df.joblib')
import statsmodels.api as sm
import pandas
from patsy import dmatrices

# R のデータセットのレポジトリから指定のデータをダウンロード
df = sm.datasets.get_rdataset("Guerry", "HistData").data

# 列の値を指定
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']

# 読み込んだデータについて、さらに上記の列データに関してデータを整形
df = df[vars]

# 読み込んだデータのうち、出力するデータの種類数を指定し、出力

# print(df[-10:])

# 欠落しているデータを除く処理
df = df.dropna()

# print(df[-5:])

# 最小二乗法
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data = df, return_type = 'dataframe')

# 最小二乗法の出力結果
# print(y[:3])
# print(X[:3])

# モデルの宣言
mod = sm.OLS(y, X)

# モデルのフィット
res = mod.fit()

# モデルのサマリーの出力
print(res.summary())

# 回帰曲線の描画
print(sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'], data = df, obs_labels =False))
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import rdata 

import warnings
warnings.filterwarnings('ignore')

# Rを使わずに直接データを成形して読み込む
parsed = rdata.parser.parse_file('../data/vouchers.rda')
converted = rdata.conversion.convert(parsed)
vouchers = converted['vouchers']

# Angrist(2002), Table 3. の再現

# 介入変数
formula_x_base = ['VOUCH0']

# 共変量
formula_x_covariate = ['SVY',  'HSVISIT', 'AGE', 'STRATA1', 'STRATA2', 'STRATA3', 'STRATA4', 
                       'STRATA5', 'STRATA6', 'STRATAMS', 'D1993', 'D1995', 'D1997',
                       'DMONTH1', 'DMONTH2', 'DMONTH3', 'DMONTH4', 'DMONTH5', 'DMONTH6', 'DMONTH7', 'DMONTH8',
                       'DMONTH9', 'DMONTH10', 'DMONTH11', 'DMONTH12', 'SEX2']

# 被説明変数
formula_ys = ['TOTSCYRS','INSCHL','PRSCH_C','USNGSCH','PRSCHA_1','FINISH6','FINISH7','FINISH8','REPT6','REPT','NREPT',
             'MARRIED','HASCHILD','HOURSUM','WORKING3']

# 回帰分析
def get_VOUVH0_regression_summary(df, formula_x_base=None, formula_x_covariate=None, formula_y=None):
    y = df[formula_y]
    if formula_x_covariate is None:
        X = df[formula_x_base]
    else:
        X = df[formula_x_base + formula_x_covariate]
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    summary = results.summary().tables[1]
    summary = pd.read_html(summary.as_html(), header=0, index_col=0)[0]
    VOUCH0_summary = summary.loc['VOUCH0']
    # 共変量を含む場合
    if formula_x_covariate is None:
        VOUCH0_summary.name = formula_y + '_base'
    # 共変量を含まない場合
    else:
        VOUCH0_summary.name = formula_y + '_covariate'
    return VOUCH0_summary

# bogota(1995)のデータを抽出
regression_data = vouchers[(vouchers.TAB3SMPL == 1) & (vouchers.BOG95SMP == 1)]

# まとめて回帰分析を実行
regression_results = []
for formula_y in formula_ys:
    # 共変量を含まない回帰
    regression_results.append(get_VOUVH0_regression_summary(
        regression_data,
        formula_x_base=formula_x_base,
        formula_x_covariate=None,
        formula_y=formula_y)
        )
    # 共変量を含む回帰
    regression_results.append(get_VOUVH0_regression_summary(
    regression_data,
    formula_x_base=formula_x_base,
    formula_x_covariate=formula_x_covariate,
    formula_y=formula_y)
    )

# 各回帰分析の結果を結合
regression_results =pd.concat(regression_results, axis=1).T

# 通学率と奨学金の利用傾向の可視化
# PRSCHA_1、USNGSCH に対するVOUCH0の効果を取り出す
# using_voucher_results = regression_results.loc[regression_results.index.str.contains('PRSCHA_1|USNGSCH', regex=True)]

# fig = px.scatter(using_voucher_results, x=using_voucher_results.index, y='coef', error_y='std err',
#                  title='2.3.3 通学と割引券の利用傾向')
# fig.show()
# fig.write_html('../images/ch2_plot2-1.html', auto_open=False)

# 留年の傾向を可視化
# PRSCH_C, INSCHL, FINISH6-8, REPT に対するVOUCH0の効果を取り出す
going_private_results = regression_results.loc[
    ['FINISH6_covariate', 'FINISH7_covariate', 'FINISH8_covariate', 'INSCHL_covariate', 'NREPT_covariate', 'PRSCH_C_covariate',   
    'REPT_covariate', 'REPT6_covariate']
]

fig = px.scatter(going_private_results, x=going_private_results.index, y='coef', error_y='std err',
                 title='2,4 留年と進級の傾向')
fig.show()
fig.write_html('../images/ch2_plot2-2.html', auto_open=False)
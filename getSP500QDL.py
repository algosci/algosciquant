

import pandas as pd
import quandl as ql
import datetime as dt

##########################################

# 3.5. Assets and Market Relationship (Systematic Risk)

# Asset/Portfolio: U.S. Large Cap Stock Market (S&P 500 Index, Vanguard VFINX Mutual Fund)
# Market: U.S. Total Stock Market (Russell 3000 Index, Vanguard VTSMX Mutual Fund)
# Risk Free Rate: U.S. Total Money Market Mean (Treasury Bills with 1 Month Maturity Annual Yield)
# Fama-French-Carhart Factors: U.S. Total Stock Market Premium (Mkt-Rf), U.S. Total Stock Market Size Premium (SMB),
#                              U.S. Total Stock Market Investment Style Premium (HML), Risk Free Rate (Rf)

# First time input YOUR API KEY.
# query = ql.get(["YAHOO/FUND_VTSMX.6","YAHOO/FUND_VFINX.6", "USTREASURY/YIELD.1", "KFRENCH/MOMENTUM_M",
#                     "KFRENCH/FACTORS_M", "RATEINF/CPI_USA"], collapse="monthly", authtoken="aZbq-NeDus7sPMkcAFs_")

# Data Query and Range Delimiting
expretquery = ql.get(["YAHOO/INDEX_GSPC.6"], collapse="daily", authtoken="aZbq-NeDus7sPMkcAFs_")
adjclosem = expretquery['1920-10-01':'2017-1-18']

df = pd.DataFrame(adjclosem)

df.columns =['S&P']

df.fillna(df.mean(),inplace=True)

df.to_csv('data.csv',date_format='%Y%m%d')
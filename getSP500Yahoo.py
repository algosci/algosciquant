#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3

import pandas as pd
import argparse

parser = argparse.ArgumentParser("get S&P historical prices from Yahoo Finance and save to CSV file")
parser.add_argument("startDate",help="S&P start download date, YYYY-MM-DD")
parser.add_argument("endDate", help="S&P end download date, YYYY-MM-DD")
parser.add_argument("outputFile", help='Optional output filename. If not specifified the output file name defauls to "ticker_startDate_endDate.csv"', default='abc.csv', nargs='?')
args = parser.parse_args()


print(args.startDate,args.endDate)

symbol = yahoo_finance.Share("^GSPC")
spdata = symbol.get_historical(args.startDate, args.endDate)


df1 = pd.DataFrame(spdata)
#print('Yahoo raw columns = ',df1.columns)
dates=df1['Date'].values
df2=df1[['Close', 'High', 'Low', 'Open','Volume']].values


dfsp=pd.DataFrame(df2, index=dates, columns = ['Close','High','Low','Open','Volume'])
dfsp.sort_index(ascending=True,inplace=True)

print(dfsp.head(5))

dfsp.fillna(dfsp.mean(),inplace=True)

if args.outputFile == 'abc.csv':
    fstr="sp500_"+args.startDate+"_"+args.endDate
else:
    fstr=args.outputFile

print("output filename = ",fstr)

dfsp.to_csv(fstr)



#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Join two time series files, where the first column is the date-time. For consistent resutls both files should use the same date-time format')
parser.add_argument("tsFile1",help="Time Series File 1")
parser.add_argument("tsFile2", help="Time Series File 2")
parser.add_argument("outputFile", help="output filename")
parser.add_argument("-j","--joinType", help='Join type, two options "inner" or "outer" (default: "outer") ',type=str, choices=["inner","outer"], default="outer")
args = parser.parse_args()

print(args.tsFile1,args.tsFile2,args.joinType)

df1 = pd.read_csv(args.tsFile1,index_col=0,parse_dates=True)
df2 = pd.read_csv(args.tsFile2,index_col=0,parse_dates=True)


df3=pd.concat([df1,df2],join=args.joinType)

grouped = df3.groupby(level=0)
df4 = grouped.last()


# df4.fillna(df4.mean(),inplace=True)
df4.fillna(method='pad', inplace=True)
print("")
#print(args.outputFile,"tail(5)")
print(df4.tail(5))

df4.to_csv(args.outputFile)



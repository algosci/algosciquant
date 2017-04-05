#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3



import argparse
import pprint
import os
from algosciquant import getIntrinioStockData
from IntrinioApiCredentials import *


pp = pprint.PrettyPrinter(indent=2)

parser = argparse.ArgumentParser("get historical stock data from Intrinio and save to CSV file")
parser.add_argument("ticker",help="ticker symbol")
parser.add_argument("startDate",help="Start date, YYYY-MM-DD")
parser.add_argument("endDate", help="End date, YYYY-MM-DD")
parser.add_argument("outputFile", help='Optional output filename. If not specifified the output file name defauls to "ticker_startDate_endDate.csv"', default='abc.csv', nargs='?')
args = parser.parse_args()

print(args.ticker,args.startDate,args.endDate)

if args.outputFile == 'abc.csv':
    outputFile="stock_"+args.ticker+"_"+args.startDate+"_"+args.endDate+".csv"
else:
    outputFile=args.outputFile

dfh=getIntrinioStockData(args.ticker,args.startDate,args.endDate,api_username,api_password,items=[],v=1)

# save CSV file
working_dir = os.path.dirname(os.path.realpath(__file__))
print("Working directory = ", working_dir, "\nOutput filename          = ", outputFile)
dfh.to_csv(outputFile)
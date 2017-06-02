#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3


import argparse
import os
from algosciquant import getGoogleStockData

parser = argparse.ArgumentParser("get ticker,stock data, historical prices from Google Finance and save to CSV file")
parser.add_argument("ticker",help="Ticker Symbol")
parser.add_argument("startDate",help="start download date, YYYY-MM-DD")
parser.add_argument("endDate", help="end download date, YYYY-MM-DD")
parser.add_argument("outputFile", help="filename to save the output", default='abc.csv', nargs='?')
args = parser.parse_args()

print(args.ticker,args.startDate,args.endDate)


dfstock=getGoogleStockData(args.ticker,args.startDate,args.endDate,v=1)

if args.outputFile == 'abc':
    outputFile=args.ticker+"_"+args.startDate+"_"+args.endDate+".csv"
else:
    outputFile=args.outputFile


# save CSV file
working_dir = os.path.dirname(os.path.realpath(__file__))
print("Working directory = ", working_dir, "\nOutput filename          = ", outputFile)

dfstock.to_csv(outputFile)
#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3


import argparse
import os
from algosciquant import getYahooSPData

parser = argparse.ArgumentParser("get ticker,f stock data, historical prices from Yahoo Finance and save to CSV file")
parser.add_argument("startDate",help="start download date, YYYY-MM-DD")
parser.add_argument("endDate", help="end download date, YYYY-MM-DD")
parser.add_argument("outputFile", help='filename to save the output (optional). Default output filename is GSPC_startDate_endDate', default='abc.csv', nargs='?')
args = parser.parse_args()


ticker='GSPC'
print(ticker,args.startDate,args.endDate)

dfstock=getYahooSPData(args.startDate,args.endDate,v=0)

if args.outputFile == 'abc.csv':
    outputFile=ticker+"_"+args.startDate+"_"+args.endDate+".csv"
else:
    outputFile=args.outputFile


# save CSV file
working_dir = os.path.dirname(os.path.realpath(__file__))
print("Working directory = ", working_dir, "\nfilename          = ", outputFile)

dfstock.to_csv(outputFile)
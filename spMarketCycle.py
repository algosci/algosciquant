#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3


import pandas as pd
import argparse
from algosciquant import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import datetime as dt

parser = argparse.ArgumentParser('S&P 500 prediction Up/Down Signal')
parser.add_argument('startTrain',help='Classifier training start date, YYYY-MM-DD')
parser.add_argument('startTest', help='Classifier test start date, YYYY-MM-DD')
parser.add_argument('endTest', help='Classifier test end date, YYYY-MM-DD')
parser.add_argument('outputFile', help='output filename',default="abc.csv",nargs='?')
args = parser.parse_args()

if args.outputFile != "abc.csv":
    outputFile = "sp500pred.csv"


print('')
print('ML Learing start training date:   ',args.startTrain)
print('Start test date:       ',args.startTest)
print('End test date:         ',args.endTest)
print('Output file:           ',args.outputFile )
print('')


train_st = dt.datetime.strptime(args.startTrain,'%Y-%m-%d')
test_st = dt.datetime.strptime(args.startTest,'%Y-%m-%d')
test_et = dt.datetime.strptime(args.endTest,'%Y-%m-%d')



#####  Import price data

print("*** load S&P data")
dfsp = pd.read_csv('sp500.csv',index_col=0,parse_dates=True)


today = dt.date.today()
if args.outputFile == 'abc.csv':
    outputFilename = 'sp_strategy_trade_summary_'+str(today)+'.csv'
else:
    outputFilename = args.outputFile


### Market Cycle

price_variable='close_price'
stmc=dt.datetime(2010,1,1)
etmc=dt.datetime(test_et.year,test_et.month,test_et.day)

print('price_variable =',price_variable)
print('start market cycle date =',stmc)
print(" ...")

mcdp = 0.01
mcup = 0.01
mudLogic=mudLogic1

df=dfsp.loc[stmc:etmc]
initMarket=1   # = 1 2010-1-1  .... = 1 2014,1,1 ... = -1, 1990,1,1
(dfmc, dfmcsummary)=marketCycle(df,initMarket ,price_variable,mcdp,mcup,mudLogic)

print("\n")
print('Market cycle summary, dfmcsummary tail(5)')
print(dfmcsummary[['mkt','startTime','endTime','startPrice','endPrice','mcnr']].tail(5))

print("\n")
print('Market cycle detail, dfmc.tail(30)')
print(dfmc[['close_price','mkt','mchlm','newmhlm','mcnr','muchp','mdclp','mcudthr','mucdown','mdcup','sdm','dfhlm','newmhlm','mcupm']].tail(30))



### Strategy Trade

pvariable='mcupm'
start_strategy_trade=dt.datetime(test_st.year,test_st.month,test_st.day)
startyr=start_strategy_trade.year
end_strategy_trade = dt.datetime(test_et.year,test_et.month,test_et.day)
endyr=test_et.year
yday = test_et.timetuple().tm_yday

df1=dfmc.loc[start_strategy_trade:end_strategy_trade,['mcupm','sdm']]
df2 = dfsp.loc[start_strategy_trade:end_strategy_trade, [price_variable]]

dft=pd.concat([df1, df2], axis=1,join='inner')
dft = tradeReturns(dft,price_variable)
dft = strategyTrade(dft,price_variable,price_variable+'_R',pvariable)

print("\n")
print("trade detail")

print(dft[[price_variable, price_variable + '_R', price_variable + '_SP', 'sdm', pvariable]].head(5))
print("...")
print(dft[[price_variable, price_variable + '_R', price_variable + '_SP', 'sdm', pvariable]].tail(10))

print("")

### BackTest

(dftsummary,dfreturns)=backTest(dft,price_variable,start_strategy_trade,end_strategy_trade)

print("backtest summary")
print(dftsummary[['start_date','end_date','start_'+price_variable,'end_'+price_variable, 'start_'+price_variable+'_SP','end_'+price_variable+'_SP' ,'return','return_SP']])
print("")
print("backtest returns summary")
print(dfreturns[[ 'nyear',  'Rc','Ra','Rc_strat','Ra_strat']])

# Save to file

dftsummary[['start_date','end_date','start_'+price_variable,'end_'+price_variable, 'start_'+price_variable+'_SP','end_'+price_variable+'_SP' ,'return','return_SP']].to_csv(outputFilename,mode = 'a')
dfreturns[[ 'nyear',  'Rc','Ra','Rc_strat','Ra_strat']].to_csv(outputFilename,mode='a')
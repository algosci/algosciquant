#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3


import pandas as pd
import argparse
from algosciquant import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import datetime as dt

parser = argparse.ArgumentParser('Stock prediction Up/Down Signal')
parser.add_argument('startTrain',help='Classifier training start date, YYYY-MM-DD')
parser.add_argument('startTest', help='Classifier test start date, YYYY-MM-DD')
parser.add_argument('endTest', help='Classifier test end date, YYYY-MM-DD')
parser.add_argument('outputFile', help='output filename',default="abc.csv",nargs='?')
args = parser.parse_args()

if args.outputFile != "abc.csv":
    outputFile = 'sp500.csv'

ticker='^GSPC'
print('')
print('ML Learing start training date:   ',args.startTrain)
print('Start test date:       ',args.startTest)
print('End test date:         ',args.endTest)
print('Output file:           ',args.outputFile )
print('')


train_st = dt.datetime.strptime(args.startTrain,'%Y-%m-%d')
test_st = dt.datetime.strptime(args.startTest,'%Y-%m-%d')
test_et = dt.datetime.strptime(args.endTest,'%Y-%m-%d')

today = dt.date.today()
if args.outputFile == 'abc.csv':
    outputFilename = 'strat_trade_summary_sp500'+str(today)+'.csv'
else:
    outputFilename = args.outputFile



#####  Import price data

dataStartDate=dt.datetime(1990,1,1)

print("*** load stock data")
read_filename='sp500.csv'
dfsp = pd.read_csv(read_filename,index_col=0,parse_dates=True)
dfsp = dfsp[dataStartDate:]

print(dfsp.head(5))
print('...)')
print(dfsp.tail(5))

### Truth  ##

print('\ntruth variable ...')

price_variable='close_price'
ndtrend=40
dx=0.01
dftruth,dfnday=ndTrendFill(dfsp,price_variable,ndtrend,dx)

print(dftruth.columns)
print('...')
dftruth['t_np1'] = dftruth['t'].shift(-1)
del dftruth['t']
print("\ndftruth columns")
print(dftruth.columns)
print(dftruth.tail(5))

ndays_dt = dt.timedelta(days=ndtrend)
plus_ndays = dfnday.index[len(dfnday.index)-2] + ndays_dt
current_cycle_startday_str=str(dfnday.index[len(dfnday.index)-2].year)+'-'+str(dfnday.index[len(dfnday.index)-2].month)+'-'+str(dfnday.index[len(dfnday.index)-2].day)

pndays_str=str(plus_ndays.year)+'-'+str(plus_ndays.month)+'-'+str(plus_ndays.day)

print('\ncurrent', ndtrend,'cylce start day = ',current_cycle_startday_str)
print('next', ndtrend,'cylce start day = ',pndays_str)

### ML Features

print('\nML features ...')
print(ticker)
f2p=[2,5,10,20,30,60,90,120]

mlFeatures = ['sp_close_price', 'sp_close_pricer', 'sp_volume', 'sp_volumer']
print('\nmlFeatures')
print(mlFeatures)
print('\ndfsp.columns')
print(dfsp.columns)
dfML = mlHistoryFeatures(dfsp, dfsp, mlFeatures, f2p)

# S&P High and Low Relative to Open
dfML['high_price_ropen'] = dfsp['high_price'] / dfsp['open_price'] - 1
dfML['low_price_ropen'] = dfsp['low_price'] / dfsp['open_price'] - 1

print("\nML features")
print(dfML.columns)

print('\nML Features (tail(3))')
print(dfML.tail(3))


#### Machine Learning

print('\n machine learning')
X = dfML.loc[train_st:test_et,dfML.columns]
Y = dftruth.loc[train_st:test_et]

print("X columns")
print(X.columns)

print('Y columns')
print(Y.columns)

model='RF'

print('ticker =',ticker)
print('ndtrend = ',ndtrend)
print('test start date =',test_st)

dfTR,clf = mClfTrainTest(X,Y,train_st,test_st,test_et,model)

### Smooth Predictions

print('\nsmooth predictions ...')
startindex = dfnday.index[dfnday.index.searchsorted(test_st)]
print('startindex =',startindex)
dfps = ndTrendSmooth(dfTR,startindex,'p',ndtrend)
add_dftr_cols=['train_st','train_et','t','p','t_np1','p_np1']
dfps[add_dftr_cols]=dfTR[add_dftr_cols]

dfps[price_variable]=dfsp[price_variable]

print(dfps[[price_variable,'train_st','train_et','t', 'p','t_np1','p_np1']].tail(5))

### Strategy Trade

print('\n strategy trade ...')
pvariable='p'   #, p, ps, mcupm
print('start date =',test_st)
print('end date =',test_et)

start_strategy_trade=test_st
startyr=start_strategy_trade.year
today = dt.date.today()
end_strategy_trade=dt.datetime(today.year,today.month,today.day)
endyr=today.year
startyr=start_strategy_trade.year
yday = datetime.today().timetuple().tm_yday

print(price_variable)
print('model =',model)
print('ndtrend =',ndtrend)

df1=dfps.loc[start_strategy_trade:end_strategy_trade,[price_variable+'r','t','p','ps','t_np1','p_np1']]

df2=dfsp.loc[start_strategy_trade:end_strategy_trade,[price_variable]]

dft=pd.concat([df1, df2], axis=1,join='inner')
dft = tradeReturns(dft,price_variable)
dft = strategyTrade(dft,price_variable,price_variable+'_R',pvariable)
dft.loc[dft.index[0],price_variable+'_SP']=dft.loc[dft.index[0],price_variable]

print(dft[[price_variable, price_variable + '_R', price_variable + '_SP', 't', pvariable, 't_np1', 'p_np1']].head(5))

print(dft[[price_variable, price_variable + '_R', price_variable + '_SP', 't', pvariable, 't_np1', 'p_np1']].tail(5))


### BackTest



print(price_variable)
print('model =',model)
print('ndtrend =',ndtrend)
(dftsummary,dfreturns)=backTest(dft,price_variable,start_strategy_trade,end_strategy_trade)

print(dftsummary)
print("")
print(dfreturns[[ 'nyear',  'Rc','Ra','Rc_strat','Ra_strat']])


print('\ncurrent', ndtrend,'cylce start day = ',current_cycle_startday_str)
print('next', ndtrend,'cylce start day = ',pndays_str)

# Save to File
#dftsummary[['start_date','end_date','start_'+price_variable,'end_'+price_variable, 'start_'+price_variable+'_SP','end_'+price_variable+'_SP' ,'return','return_SP']].to_csv(outputFilename,mode = 'a')
#dfreturns[[ 'nyear',  'Rc','Ra','Rc_strat','Ra_strat']].to_csv(outputFilename,mode='a')
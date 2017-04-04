# Quant Functions

import pandas as pd
import numpy as np
import scipy
from datetime import datetime

import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.preprocessing import Imputer


def getYahooStockData(ticker,startDate,endDate,v = 0):
    import yahoo_finance
    symbol = yahoo_finance.Share(ticker)
    import sys

    try:
        stockData = symbol.get_historical(startDate,endDate)
        df1 = pd.DataFrame(stockData)
        dates=df1['Date'].values
        df2=df1[['Adj_Close','Close', 'High', 'Low', 'Open', 'Volume']].values
        dfs = pd.DataFrame(df2, index=dates, columns=['Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
        dfs.sort_index(ascending=True, inplace=True)
    except:
        print('Exception: Double check your ticker, startDate and endDate and make sure there is data available.')
        print('ticker = ',ticker)
        print('startDate = ',startDate)
        print('endDate =',endDate)
        e = sys.exc_info()[0]
        print(e)


    dfs.fillna(method='pad', inplace=True)

    # Ensure all columns are numeric (sometimes returns from Yahoo are interpreted as text)
    dfs['Adj_Close'] = pd.to_numeric(dfs['Adj_Close'])
    dfs['Open'] = pd.to_numeric(dfs['Open'])
    dfs['Close'] = pd.to_numeric(dfs['Close'])
    dfs['High'] = pd.to_numeric(dfs['High'])
    dfs['Low'] = pd.to_numeric(dfs['Low'])
    dfs['Volume'] = pd.to_numeric(dfs['Volume'])
    dfs.index = pd.to_datetime(dfs.index)

    # Normalize column names (same as Intrinio column names)
    dfs.rename(columns={ 'Adj_Close':'adj_close_price' ,'Open': 'open_price', 'Close': 'close_price', 'High': 'high_price', 'Low': 'low_price', 'Volume': 'volume'},inplace=True)

    if v > 0:
        print(dfs.head(5))

    return dfs

def getYahooSPData(startDate,endDate,v = 0):

    dfsp=getYahooStockData('^GSPC',startDate,endDate,v=0)
    del dfsp['adj_close_price']

    if v > 0:
        print(dfsp.head(5))

    return dfsp



def getIntrinioStockData(ticker,startDate,endDate,api_username,api_password,items=[],v=1):
    import requests
    from io import StringIO
    import sys

    items2 = [ 'close_price',  'adj_close_price',  'adj_high_price',  'adj_low_price', 'adj_open_price',
             'adj_volume',  'high_price', 'low_price', 'open_price',  'volume',
              'pricetoearnings', 'ebitda',  'ebitdagrowth',  'earningsyield',
              'quickratio', 'pricetobook',  'evtoebitda', 'evtorevenue']

    if items:
        items2=items
    historical_data_url = 'https://api.intrinio.com/historical_data'
    qTicker = "?ticker=" + ticker
    atStartDate = "&start_date=" + startDate
    atEndDate = "&end_date=" + endDate

    # Ticker Financial Parameters
    k = 1
    for item in items2:
        atItem = "&item=" + item
        rstr = historical_data_url + qTicker + atItem + atStartDate + atEndDate
        p = {'ticker': ticker, 'item': item, 'start_date': startDate, 'end_date': endDate}
        if v > 0:
            print(ticker, item)
            print(rstr)

        try:
            # r = requests.get(rstr,auth=(api_username, api_password))
            r = requests.get(historical_data_url, params=p, auth=(api_username, api_password))
            # nested JSON
            x = pd.read_json(StringIO(r.text), orient='columns')
            # flatten JSON
            dfh2 = pd.read_json(x['data'].to_json(), orient='index')
            dfh2.set_index('date', inplace=True)
            dfh2.index = pd.to_datetime(dfh2.index)
            dfh2.rename(columns={"value": item}, inplace=True)
            dfh2.sort_index(ascending=True, inplace=True)
            if v > 0:
                print(r)
            if k > 1:
                #print(dfh.shape,dfh2.shape)
                dfh = dfh.join(dfh2)

            else:
                dfh = dfh2

            #print('k = ',k)
            k += 1
            dfh.fillna(method='pad', inplace=True)

        except TypeError:
            e = sys.exc_info()[0]
            print("Exception occurred, TypeError")
            print("  There is likely no data for request", rstr)

            pass

            #print(e)
            #raise
        except KeyError:
            e = sys.exc_info()[0]
            print("Exception occurred. KeyError")
            print("There is likely no data for request", rstr)
            pass

    return dfh


def ndTrendFill(dfs,price_variable,ndays):
    df=dfs
    ndays_dt = dt.timedelta(days=ndays)
    dfndf=pd.DataFrame(index=dfs.index)

    first_i = df.index[0]
    last_i = df.index[len(df.index)-1]
    fi_plus_ndays = first_i + ndays_dt
    lastindex = len(df.index)-1
    last_i = df.index[lastindex]
    nextindex=df.index.searchsorted(dt.datetime(fi_plus_ndays.year,fi_plus_ndays.month, fi_plus_ndays.day))
    next_i=df.index[nextindex]
    #print('first_i = ',first_i,'next_i = ',next_i,'last_i = ',last_i)
    prev_i = first_i
    previndex=0
    while next_i < last_i:
        i_plus_ndays = prev_i + ndays_dt
        nextindex=df.index.searchsorted(dt.datetime(i_plus_ndays.year,i_plus_ndays.month, i_plus_ndays.day))
        if nextindex > lastindex:
            nextindex = lastindex
        next_i=df.index[nextindex]
        #if (nextindex == lastindex) or (prev_i == first_i):
        #print('prev_i = ',prev_i,'next_i = ',next_i,'t = ',t)
        t=np.sign( df.loc[df.index[nextindex],price_variable]-df.loc[df.index[previndex],price_variable])
        if t ==0:
            t = -1
        #print(next_i,last_i,nextindex,lastindex,df.index[nextindex],df.index[previndex],t)
        for index in range(previndex,nextindex):
            dfndf.loc[df.index[index],'t']=t
        if nextindex == lastindex:
            #print('last_price = ',df.loc[df.index[nextindex],price_variable],'prev_price = ',df.loc[df.index[previndex],price_variable] )
            dfndf.loc[last_i] = t
        prev_i = next_i
        previndex = nextindex
    return dfndf




def getIntrinioSPData(startDate,endDate,api_username,api_password,items=[],v=1):
    #https://api.intrinio.com/historical_data?ticker=$SPX&item=open_price&start_date=2007-01-01&end_date=2017-02-17
    ticker = "$SPX"

    items2 = [ 'close_price',  'high_price',  'low_price',  'open_price',  'volume']

    if items:
        items2=items

    dfh = getIntrinioStockData(ticker, startDate, endDate, api_username, api_password, items2, v=v)

    return dfh



# Trade Returns
#  computes returns
#   Input
#       DataFrame indexed by time sorted in ascending order
#       Columns must contain "P" Price Column
#           P := price column
#   Output
#       DataFrame same as input DataFrame + one additional column
#       Columns (in addition to input columns)
#           R := returns column
def tradeReturns(df,ticker):
    tickerReturns=ticker + "_R"
    r=df[ticker] / df[ticker].shift(1) - 1
    df.loc[:,tickerReturns]=r
    df.loc[df.index[0],tickerReturns]=0
    return  df


# Hold or Sit trading strategy
#  Inputs
#     DataFrame
#        ticker := price column
#        returns := market returns column
#        tradeHold := column with trade or hold signal (1 := trade, 0 := hold, do not trade)
#  Outputs
#     DataFrame
#        ticker_SP : Strategy Price
def strategyTrade(df,ticker,returns,tradeHold):

    # Strategy Price Column name
    strategyPrice = ticker+"_SP"

    # New column Strategy Price
    df.loc[:, strategyPrice] = [0] * df.index.size

    # initialize holdPrice
    holdPrice = df.loc[df.index[0], ticker]

    # loop through df and apply trading strategy
    for i in df.index:
        if df.loc[i, tradeHold] == 1:
            # compute new strategy price
            SP=holdPrice * (1 + df.loc[i, returns])
            df.loc[i, strategyPrice] = SP
            # store new holdPrice
            holdPrice = SP
        elif df.loc[i,tradeHold] == 0 or df.loc[i,tradeHold]== -1:
            df.loc[i, strategyPrice] = holdPrice
    # Return DF
    return df



#################################################################
#  Confustion Matrix
#    Market Cycle Classification
#
def clfMktConfusionMatrix(df,truth,classified):


    err = df[df[truth]!= df[classified]]
    corr = df[df[truth] == df[classified]]
    neg = df[df[truth] == 1].index.size
    pos = df[df[truth] == -1].index.size
    samplesize = df.index.size
    errors = err.index.size
    correct = corr.index.size
    er = (1.0*errors)/(1.0*samplesize)

    print('total =', samplesize, '\n  pos (up) = ', pos,'\n  neg (dwn)',neg)
    print('errors =',errors,'correct = ',correct)

    fn = err[err[classified] == 1].index.size
    fp = err[err[classified] == -1].index.size
    tp = corr[corr[truth] == -1].index.size
    tn = corr[corr[truth] == 1].index.size
    if pos !=0:
        tpr = (1.0 * tp) / (1.0 * pos)
        fnr = (1.0 * fn) / (1.0 * pos)
    else:
        tpr = 0
        fnr = 0
    if neg != 0:
        fpr = (1.0 * fp) / (1.0 * neg)
        tnr = (1.0 * tn) / (1.0 * neg)
    else:
        fpr = 0
        tnr = 0

    dfCMdef=pd.DataFrame({'Predicted MktDwn':['tp','fp'],'Predicted MktUp':['fn','tn'],'Actual Totals':['pos:MktDwn', 'neg:MktUp']}, index=['actual MktDwn','actual MktUp'])
    dfCMA=pd.DataFrame({'Predicted MktDwn':[tp,fp],'Predicted MktUp':[fn,tn],'Actual Totals':[pos, neg]}, index=['actual MktDwn','actual MktUp'])
    dfCMR=pd.DataFrame({'Predicted MktDwn':[tpr,fpr],'Predicted MktUp':[fnr,tnr],'Actual Totals': [pos, neg]}, index=['actual MktDwn','actual MktUp'])

    return samplesize, errors, correct, er, fn, fp, tp, tn, fnr, fpr, tpr, tnr, dfCMdef, dfCMA, dfCMR

### Machine Learning Feature-Predictors DataFrame
# Inputs
#    dfstk := Stock-Security data frame ... Intrinio compatible column names ... e.g. adj_close_price, adj_volume
#    dfspi := S&P or other Market Index Data Frame ... Intrinio compatible column names, e.g. close_price, volume
#    mlFeatures .:= optional list. Default mlFeatures=['adj_close_pricer', 'adj_volumer', 'pricetoearnings', 'sp_close_pricer']
# Outputs
#    dfML := DataFrame of Predictors derived from Features (1, 2, 3 day history, 10, 30, 60 day moving averages
#
def mlHistoryFeatures(dfstk,dfsp,mlFeatures=['adj_close_pricer', 'adj_volumer', 'pricetoearnings', 'sp_close_pricer','sp_volumer'],f2p=[3,5,10,30,90,120]):
    tvariable = mlFeatures[0]


    # SP ML Features
    dfML = dfstk[[]].copy()

    stockItems = dfstk.columns
    spItems = dfsp.columns
    stockFeatures = []
    spFeatures = []
    for k in mlFeatures:
        if k in stockItems:
          stockFeatures.append(k)
        if k in spItems:
            spFeatures.append(k)


    if (len(stockItems) == 0 and len(spItems) == 0):
        print("Error: no data frame features selected")

    for key in mlFeatures:
        if 'sp_close_pricer' in mlFeatures:
            dfML['sp_close_pricer'] = dfsp['close_price'] / dfsp['close_price'].shift(1) - 1

        if 'sp_volumer' in mlFeatures:
            dfML['sp_volumer'] = dfsp['volume'] / dfsp['volume'].shift(1) - 1

        if 'adj_close_pricer' in mlFeatures:
            dfML['adj_close_pricer'] = (dfstk['adj_close_price']) / dfstk['adj_close_price'].shift(1) - 1

        if 'adj_volumer' in mlFeatures:
            dfML['adj_volumer'] = (dfstk['adj_volume']) / dfstk['adj_volume'].shift(1) - 1

        if key in stockFeatures:
            if key not in ['adj_close_pricer', 'adj_volumer']:
                dfML[key] = pd.to_numeric(dfstk[key])

        if key in spFeatures:
            if key not in ['sp_close_pricer', 'sp_volumer']:
                dfML[key] = pd.to_numeric(dfsp[key])

    # There are some days that no S&P data is available. Fill these with the previous day or it will mess up Machine Learning
    dfML.fillna(method='pad', inplace=True)

    # Feature to Predictor Variables

    for mlFeature in mlFeatures:
        x = mlFeature

        # N-day net price change
        for i in range(1, f2p[0] + 1):
            dfML[x + '_h' + str(i)] = dfML[x].shift(i)

        # N-day Moving averages
        for i in f2p[1:len(f2p) + 1]:
            period = i
            dfML[x + '_ma' + str(i)] = (dfML[x].rolling(center=False, min_periods=period, window=period).sum() / period)


    return dfML


def backTest(dft,price_variable,start_date,end_date):

    start_strategy_trade=dt.datetime(start_date.year,start_date.month,start_date.day)
    end_strategy_trade=dt.datetime(end_date.year,end_date.month,end_date.day)
    startyr=start_strategy_trade.year
    endyr=end_strategy_trade.year
    dftsummary = pd.DataFrame(index=range(startyr, endyr))



    last_ix = (len(dft))
    for year in range(startyr, endyr + 1):

        start_ix = dft.index.searchsorted(dt.datetime(year, 1, 1))
        if year != endyr:
            end_ix = dft.index.searchsorted(dt.datetime(year, 12, 31))
        else:
            end_ix = dft.index.searchsorted(dt.datetime(end_strategy_trade.year, end_strategy_trade.month, end_strategy_trade.day))

        six = start_ix
        eix = end_ix
        if eix == last_ix:
            eix = end_ix - 1
        dftsummary.loc[year, 'start_date'] = dft.index[six]
        dftsummary.loc[year, 'start_' + price_variable] = dft.ix[six, price_variable]
        dftsummary.loc[year, 'end_date'] = dft.index[eix]
        dftsummary.loc[year, 'end_' + price_variable] = dft.ix[eix, price_variable]

        if year == startyr:
            dftsummary.loc[year, 'start_' + price_variable + '_SP'] = dft.ix[six, price_variable]
        else:
            dftsummary.loc[year, 'start_' + price_variable + '_SP'] = dftsummary.ix[
                last_year, 'end_' + price_variable + '_SP']
        dftsummary.loc[year, 'end_' + price_variable + '_SP'] = dft.ix[eix, price_variable + '_SP']
        dftsummary.loc[year, 'return'] = dftsummary.loc[year, 'end_' + price_variable] / dftsummary.loc[
            year, 'start_' + price_variable] - 1
        dftsummary.loc[year, 'return_SP'] = dftsummary.loc[year, 'end_' + price_variable + '_SP'] / dftsummary.loc[
            year, 'start_' + price_variable + '_SP'] - 1
        last_year = year


    n = (end_strategy_trade - start_strategy_trade).days / (365)
    Rc = dftsummary.loc[endyr, 'end_' + price_variable] / dftsummary.loc[startyr, 'start_' + price_variable] - 1
    Rc_strat = dftsummary.loc[endyr, 'end_' + price_variable + '_SP'] / dftsummary.loc[
        startyr, 'start_' + price_variable + '_SP'] - 1
    Ra = ((Rc + 1) ** (1 / n)) - 1
    Ra_strat = ((Rc_strat + 1) ** (1 / n)) - 1


    d={ 'nyear':n,  'Ra': Ra,'Ra_strat': Ra_strat,'Rc': Rc,'Rc_strat': Rc_strat}
    dfreturns=pd.DataFrame(d,index=[' '])


    return dftsummary, dfreturns

def mClfTrainTest(X, Y, train_st, test_st, test_et, clf, trainndays = -1,mc=0,dftflag=pd.DataFrame()):
    import sys
    year = train_st.year
    month = train_st.month

    dfTR = X.loc[test_st:test_et, X.columns]
    dfTR['t_np1'] = Y.loc[test_st:test_et]
    if mc==1:
        dfTR['tf']=dftflag.loc[test_st:test_et] #train flag 1 = train model, 0 = don't train, use previously trained model
    else:
        dfTR['tf']=[1]*len(dfTR.index)

    print("last t_np1 = ", Y.index[len(Y)-1], Y.loc[Y.index[len(Y)-1], ])
    #  classifier prediction column
    dfTR.loc[:, 'p_np1'] = [0] * dfTR.index.size

    # Each prediction is for one day in the future

    llast_i=dfTR.index[0]
    last_i=llast_i
    for i in dfTR.index:

        train_st2=train_st
        ndaydt = dt.timedelta(days=trainndays)
        tmp_st=i - ndaydt
        if tmp_st > train_st:
            train_st2=tmp_st

        dfTR.loc[i, 'train_st'] = train_st2


        X2 = X.loc[train_st2:llast_i]
        Y2 = Y.loc[train_st2:llast_i]


        if i == dfTR.index[0]: # first time do not have a trained model
            mp = -1
            clf.fit(X2.as_matrix(), Y2.values.ravel())
            train_et=llast_i

        else:
            mp = clf.predict([X.loc[i]])
            dfTR.loc[i, 'p_np1'] = mp


        if dfTR.loc[i,'tf']==1:
            #if pd.notnull(Y2.loc[llast_i, 't_np1']):
            train_et=llast_i
            try:
                clf.fit(X2.as_matrix(), Y2.values.ravel())
            except ValueError:
                e = sys.exc_info()[0]
                print("Exception occurred, ValueError")
                print(i,last_i,llast_i)
                #print(Y2)
                #print(X2)
                pass

        dfTR.loc[i, 'train_et'] = train_et


        #if i.year != year or i.month != month:
        if i.year != year:
            print (i.strftime('%Y-%m-%d'))

            # next loop variables
        year = i.year
        month = i.month
        ## note, have to go back two days so that future truth (+1 day) data does not pollute the training and prediction for current day
        llast_i=last_i
        last_i=i


    dfTR['t']= dfTR['t_np1'].shift(1)
    dfTR['p']= dfTR['p_np1'].shift(1)
    print ("Training month", i.strftime('%Y-%m-%d'), "train_st = ",train_st, 'train_st2 = ',train_st2)
    return dfTR, clf
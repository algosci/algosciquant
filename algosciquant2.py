# Quant Functions

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import scipy
from datetime import datetime

import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
#from xgboost.sklearn import XGBClassifier
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


def getSPData(startDate,endDate,v = 0):

    dfsp=getYahooStockData('^GSPC',startDate,endDate,v=0)
    del dfsp['adj_close_price']

    if v > 0:
        print(dfsp.head(5))

    return dfsp


def dftn_from_files(sdate,edate,files,price_variable,lnames):
    k=0
    lnlist=[]
    for k in range(0,len(lnames)):
        if k==0:
            df= pd.read_csv(files[k],index_col=0,parse_dates=True)
            dftmp = pd.DataFrame(df.loc[sdate:edate, [price_variable]])
            dftn_i0 = dftmp.index[0]
            dftn=pd.DataFrame(dftmp.loc[sdate:edate,price_variable]/dftmp.loc[dftn_i0,price_variable])
            dftn_i0 = dftn.index[0]
            dftn.rename(columns={price_variable:lnames[k]},inplace=True)
        else:
            #print(files[k])
            df=pd.read_csv(files[k],index_col=0,parse_dates=True)
            #print(df.columns)
            dftn[lnames[k]]=df.loc[sdate:edate,price_variable+'_SP']/df.loc[dftn_i0,price_variable+'_SP']
        lnlist.append(lnames[k])
        k+=1
    lnames = lnlist
    return dftn, lnames


def ndayTruth(df,nday,tvariable='adj_close_price'):
    #dx=0.00001 # note, in lambda can check if x > dx, however works best with x > 0
    dfTruth=pd.DataFrame(df[tvariable])
    dfTruth[tvariable + '_n']=dfTruth[tvariable].shift(-nday)
    dfTruth['t_n']=dfTruth[tvariable+'_n']/dfTruth[tvariable]-1
    dfTruth['t_n'] = dfTruth['t_n'].apply(lambda x: 1 if x > 0  else -1)
    del dfTruth[tvariable+'_n']
    return dfTruth


def ndTrendFill(dfs,price_variable,ndays,dx):
    df=dfs
    ndays_dt = dt.timedelta(days=ndays)
    dfndf=pd.DataFrame(index=dfs.index)

    first_i = df.index[0]
    fi_plus_ndays = first_i + ndays_dt
    lastindex = len(df.index)-1
    last_i = df.index[lastindex]
    nextindex=df.index.searchsorted(dt.datetime(fi_plus_ndays.year,fi_plus_ndays.month, fi_plus_ndays.day))
    next_i=df.index[nextindex]
    #print('first_i = ',first_i,'next_i = ',next_i,'last_i = ',last_i)
    prev_i = first_i
    previndex=0

    x={df.loc[df.index[previndex],price_variable]}

    #ndindex=pd.DataFrame(data=x, index=[df.index[previndex]])
    dfnday = pd.DataFrame({price_variable:x}, index=[df.index[previndex]])


    while next_i < last_i:
        i_plus_ndays = prev_i + ndays_dt
        nextindex=df.index.searchsorted(dt.datetime(i_plus_ndays.year,i_plus_ndays.month, i_plus_ndays.day))
        tf=1

        if nextindex > lastindex:
            nextindex = lastindex
            tf=-1

        #print('previndex = ', df.index[previndex], 'nextindex =', df.index[nextindex])

        x = df.loc[df.index[nextindex],price_variable]
        dftmp = pd.DataFrame({price_variable:x},index=[df.index[nextindex]])
        dfnday = dfnday.append(dftmp)

        next_i = df.index[nextindex]
            #if (nextindex == lastindex) or (prev_i == first_i):
            #print('prev_i = ',prev_i,'next_i = ',next_i,'t = ',t)
            #t=np.sign( df.loc[df.index[nextindex],price_variable]-df.loc[df.index[previndex],price_variable])

        x=df.loc[df.index[nextindex],price_variable]/df.loc[df.index[previndex],price_variable] - 1
        if x >= dx:
            t=1
        else:
            t=-1


        for index in range(previndex,nextindex):
            if tf==1:
                dfndf.loc[df.index[index],'t']=t
            if index==previndex:
                dfndf.loc[df.index[index], 'tf'] = tf
            else:
                dfndf.loc[df.index[index], 'tf'] = -1

            # print(next_i,last_i,nextindex,lastindex,df.index[nextindex],df.index[previndex],t)

        #if nextindex == lastindex:
        #    #print('last_price = ',df.loc[df.index[nextindex],price_variable],'prev_price = ',df.loc[df.index[previndex],price_variable] )
        #    dfndf.loc[last_i] = t
        prev_i = next_i
        previndex = nextindex
    return dfndf,dfnday

def ndTrendSmooth(dfp,startindex,pvariable,ndays):
    df=dfp
    ndays_dt = dt.timedelta(days=ndays)

    dfps=pd.DataFrame(index=dfp.index)

    first_i = dfp.index[dfp.index.searchsorted(startindex)]

    last_i = df.index[len(df.index)-1]
    fi_plus_ndays = first_i + ndays_dt
    lastindex = len(df.index)-1
    last_i = df.index[lastindex]
    nextindex=df.index.searchsorted(dt.datetime(fi_plus_ndays.year,fi_plus_ndays.month, fi_plus_ndays.day))
    next_i=df.index[nextindex]
    #print('first_i = ',first_i,'next_i = ',next_i,'last_i = ',last_i)
    prev_i = first_i

    previndex=dfp.index.searchsorted(startindex)

    #initialize to p for first few indexes ... note, need to match up the nday intervals
    for index in range(0,previndex):
        dfps.loc[df.index[index],'ps']=df.loc[df.index[index],'p']

    while next_i < last_i:
        i_plus_ndays = prev_i + ndays_dt
        nextindex=df.index.searchsorted(dt.datetime(i_plus_ndays.year,i_plus_ndays.month, i_plus_ndays.day))
        if nextindex > lastindex:
            nextindex = lastindex
        next_i=df.index[nextindex]

        #print('previndex = ',df.index[previndex],'nextindex =', df.index[nextindex])
        ps_sum=0

        for index in range(previndex,nextindex):
            tmp = df.loc[df.index[index],pvariable]

            ps_sum += tmp if pd.notnull(tmp) else ps_sum
            ps = 1 if  ps_sum > 0 else -1

            dfps.loc[df.index[index], 'ps'] = ps


            #print('   ', df.loc[df.index[index],'t'], df.loc[df.index[index],'p'] , dfps.loc[df.index[index],'ps'],ps_sum )

        if nextindex == lastindex:
            #print('last_price = ',df.loc[df.index[nextindex],price_variable],'prev_price = ',df.loc[df.index[previndex],price_variable] )
            dfps.loc[last_i] = np.sign(ps)

        prev_i = next_i
        previndex = nextindex
    return dfps


def ndTrendFill2(dfs,price_variable,ndays):
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


# Strategy Trade
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
        else:

            df.loc[i, strategyPrice] = holdPrice
    # Return DF
    return df

def volatilityPrice(dfTR,vol='50',ma='120',variable=''):
    dfTR2=dfTR.copy()
    def fpv(row):
        return row['p_1'] if row['vol_y_'+vol] > 0.2 and row['ma'+ma] < 0 else 1

    def fv(row):
        return -1 if row['vol_y_'+vol] > 0.2 and row['ma'+ma] < 0 else 1

    def fpma(row):

        return row['p_1'] if row['ma'+ma] < 0 else 1

    dfTR2['pv_1'] = dfTR2.apply(fpv, axis=1)
    dfTR2['v_1'] = dfTR2.apply(fv, axis=1)
    dfTR2['pma_1'] = dfTR2.apply(fpma, axis=1)
    dfTR2['pv'] = dfTR2['pv_1'].shift(1)
    dfTR2['v'] = dfTR2['v_1'].shift(1)
    dfTR2['pma'] = dfTR2['pma_1'].shift(1)

    return dfTR2

def volatilityPriceSP(dfTR,vlty='120',ma='60',mcvariable='',vlty_thr=0.2):

    dfTR2=dfTR.copy()

    def fmcpv(row):
        return row['p_1'] if row['vol_y_' + vlty] > vlty_thr and row['close_pricer_ma' + ma] < 0 else row['mc2']

    def fmcv(row):
        return -1 if row['vol_y_' + vlty] > vlty_thr and row['close_pricer_ma' + ma] < 0 else row['mc2']

    def fpv(row):
        return row['p_1'] if row['vol_y_'+ vlty] > vlty_thr and row['close_pricer_ma'+ma] < 0 else 1

    def fv(row):
        return -1 if row['vol_y_'+ vlty] > vlty_thr and row['close_pricer_ma'+ma] < 0 else 1

    def fp(row):
        return row['p_1'] if row[mcvariable] < 1 else 1

    if mcvariable:
        dfTR2['mc2'] = dfTR[mcvariable].shift(-1)
        dfTR2[mcvariable+'v_1'] = dfTR2.apply(fmcv, axis=1)
        dfTR2[mcvariable + 'pv_1'] = dfTR2.apply(fmcpv, axis=1)
        dfTR2[mcvariable + 'p_1'] = dfTR2.apply(fmcpv, axis=1)
    dfTR2['pv_1'] = dfTR2.apply(fpv, axis=1)
    dfTR2['v_1'] = dfTR2.apply(fv, axis=1)
    dfTR2['pv'] = dfTR2['pv_1'].shift(1)
    dfTR2['v'] = dfTR2['v_1'].shift(1)
    dfTR2[mcvariable+'v'] = dfTR2[mcvariable+'v_1'].shift(1)
    dfTR2[mcvariable + 'pv'] = dfTR2[mcvariable + 'pv_1'].shift(1)
    dfTR2[mcvariable + 'p'] = dfTR2[mcvariable + 'p_1'].shift(1)
    del dfTR2['mc2']

    return dfTR2

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

    #print('total =', samplesize, '\n  pos (up) = ', pos,'\n  neg (dwn)',neg)
    #print('errors =',errors,'correct = ',correct)

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

    dfCMdef=pd.DataFrame({'Predicted MktDown':['tp','fp'],'Predicted MktUp':['fn','tn'],'Totals':['pos:MktDwn', 'neg:MktUp']}, index=['actual MktDown','actual MktUp'])
    dfCMA=pd.DataFrame({'Predicted MktDown':[tp,fp],'Predicted MktUp':[fn,tn],'Totals':[pos, neg]}, index=['actual MktDown','actual MktUp'])
    dfCMR=pd.DataFrame({'Predicted MktDown':[tpr,fpr],'Predicted MktUp':[fnr,tnr],'Totals': [pos, neg]}, index=['actual MktDown','actual MktUp'])

    return samplesize, errors, correct, er, fn, fp, tp, tn, fnr, fpr, tpr, tnr, dfCMdef, dfCMA, dfCMR

def mlStockFeatures(dfs,dfsp,dataStartDate,test_et,f=1,h=1,ma=1,v=1,sp=1):

    # Price and Volumne
    dfML = pd.DataFrame(dfs.loc[dataStartDate:test_et, ['adj_close_price']])
    dfML['volume'] = dfsp.loc[dataStartDate:test_et, 'volume']
    dfML['adj_close_pricer'] = (dfs.loc[dataStartDate:test_et, 'adj_close_price'] / dfs.loc[dataStartDate:test_et,'adj_close_price'].shift(1))-1

    #S&P
    if sp==1:
        dfML['sp_close_price'] = dfsp.loc[dataStartDate:test_et, 'close_price']
        dfML['sp_close_pricer'] = (dfML['sp_close_price'] / dfML['sp_close_price'].shift(1)) - 1
        #dfML['sp_volume'] = dfsp.loc[dataStartDate:test_et, 'volume']
        #period = 60
        #dfML['spma' + str(period)] = dfML['sp_close_pricer'].rolling(center=False, min_periods=period,window=period).sum() / period
        #period = 120
        #dfML['spma' + str(period)] = dfML['sp_close_pricer'].rolling(center=False, min_periods=period,window=period).sum() / period

    # Fundamentals
    if f == 1:
        dfML['ebitdagrowth'] = dfs.loc[dataStartDate:test_et, 'ebitdagrowth']
        dfML['pricetoearnings'] = dfs.loc[dataStartDate:test_et, 'pricetoearnings']
        dfML['quickratio'] = dfs.loc[dataStartDate:test_et, 'quickratio']

    # History
    if h > 0:
        dfML['adj_close_price_h1'] = dfML['adj_close_price'].shift(1)
        dfML['adj_close_price_h2'] = dfML['adj_close_price'].shift(2)
    if h > 1:
        dfML['adj_close_pricer_h1'] = dfML['adj_close_pricer'].shift(1)
        dfML['adj_close_pricer_h2'] = dfML['adj_close_pricer'].shift(2)

    # Moving Average Relative Price

    if ma==1:

        period = 60
        dfML['ma60'] =  dfML['adj_close_pricer'].rolling(center=False, min_periods=period, window=period).sum() / period
        period = 120
        dfML['ma120'] =dfML['adj_close_pricer'].rolling(center=False, min_periods=period, window=period).sum() / period

        period = 60
        dfML['pma60'] = dfML['adj_close_price'].rolling(center=False, min_periods=period, window=period).sum() / period
        period = 120
        dfML['pma120'] = dfML['adj_close_price'].rolling(center=False, min_periods=period, window=period).sum() / period

    # Volatility

    if v==1:
        period = 10
        dfML['vol_y_10'] = np.sqrt(252) * dfML['adj_close_pricer'].rolling(center=False, min_periods=period,  window=period).std()
        period = 50
        dfML['vol_y_50'] = np.sqrt(252) * dfML['adj_close_pricer'].rolling(center=False, min_periods=period,window=period).std()
        period = 120
        dfML['vol_y_120'] = np.sqrt(252) * dfML['adj_close_pricer'].rolling(center=False, min_periods=period,window=period).std()

    # Fill in NAs ... sometimes mismatches in dates between SP and Stock data
    dfML.fillna(method='pad', inplace=True)

    return dfML


### Machine Learning Feature-Predictors DataFrame
# Inputs
#    dfstk := Stock-Security data frame ... Intrinio compatible column names ... e.g. adj_close_price, adj_volume
#    dfspi := S&P or other Market Index Data Frame ... Intrinio compatible column names, e.g. close_price, volume
#    mlFeatures .:= optional list. Default mlFeatures=['adj_close_pricer', 'adj_volumer', 'pricetoearnings', 'sp_close_pricer']
# Outputs
#    dfML := DataFrame of Predictors derived from Features (1, 2, 3 day history, 10, 30, 60 day moving averages
#
def mlSPFeatures(dfsp,HMAFeatures=['close_pricer','volumer'],f2p=[2,5,10,20,30,60,90,120]):
    tvariable = HMAFeatures[0]

    # SP ML Features
    dfML = pd.DataFrame(index=dfsp.index)

    spFeatures= dfsp.columns

    for key in HMAFeatures:
        if 'close_pricer' in HMAFeatures:
            dfML['close_pricer'] = dfsp['close_price'] / dfsp['close_price'].shift(1) - 1

        if 'volumer' in HMAFeatures:
            dfML['volumer'] = dfsp['volume'] / dfsp['volume'].shift(1) - 1

    for key in spFeatures:
        if key not in ['close_pricer', 'volumer']:
            dfML[key] = pd.to_numeric(dfsp[key])

    # There are some days that no S&P data is available. Fill these with the previous day or it will mess up Machine Learning
    dfML.fillna(method='pad', inplace=True)

    # History and Moving Averages
    for mlFeature in HMAFeatures:
        x = mlFeature

        # Net price change
        for i in range(1, f2p[0] + 1):
            dfML[x + '_h' + str(i)] = dfML[x].shift(i)

        # N-day Moving averages
        for i in f2p[1:len(f2p) + 1]:
            period = i
            dfML[x + '_ma' + str(i)] = (dfML[x].rolling(center=False, min_periods=period, window=period).sum() / period)



    #  High and Low Relative to Open
    dfML['high_price_ropen'] = dfsp[ 'high_price'] / dfsp['open_price'] - 1
    dfML['low_price_ropen'] = dfsp[ 'low_price'] / dfsp['open_price'] - 1

    return dfML



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

        if 'sp_close_price' in mlFeatures:
            dfML['sp_close_price'] = dfsp['close_price']

        if 'sp_volumer' in mlFeatures:
            dfML['sp_volumer'] = dfsp['volume'] / dfsp['volume'].shift(1) - 1

        if 'sp_volume' in mlFeatures:
            dfML['sp_volume'] = dfsp['volume']

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


def backTestSummary(dft,price_variable,start_date,end_date):

    start_strategy_trade=dt.datetime(start_date.year,start_date.month,start_date.day)
    end_strategy_trade=dt.datetime(end_date.year,end_date.month,end_date.day)
    startyr=start_strategy_trade.year

    endyr=dft.index[len(dft.index)-1].year
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
    dfreturns=pd.DataFrame(d,index=[end_date])


    return dftsummary, dfreturns

def mClfTrainTest(X, Y, nday, train_st, test_st, test_et, model='DT', trainndays = -1,mc=0,dftflag=pd.DataFrame(),v=1):

    import sys

    # Decision Tree
    if model=='DT':
        trainndays = 400




        clf = DecisionTreeClassifier(min_samples_split=20, random_state=99, )

    # Random Forest
    elif model=='RF':
        ne=366
        trainndays = 1100

        clf = RandomForestClassifier(n_estimators=ne, random_state=2)

    # K Nearest Neighbor
    elif model=='KNN':
        k=5
        trainndays=500

        clf=KNeighborsClassifier(n_neighbors=k,algorithm='auto')

    # XG Boost
    elif model == 'XGB':
        trainndays=1000

        clf = XGBClassifier()

    # Support Vector Machine
    elif model == 'SVM':
        trainndays = 2000

        clf = svm.SVC(kernel='rbf', C=1,gamma='auto')

    # Naive-Bayes
    elif model == 'NB':
        trainndays = 1100

        clf = GaussianNB()

    ##### Logistic Regression
    elif model == 'LR':
        trainndays = 1100

        clf = LogisticRegression()

    year = train_st.year
    month = train_st.month

    dfTR = X.loc[test_st:test_et, X.columns]
    dfTR['t_n'] = Y.loc[test_st:test_et,['t_n']]
    if mc==1:
        dfTR['tf']=dftflag.loc[test_st:test_et] #train flag 1 = train model, 0 = don't train, use previously trained model
    else:
        dfTR['tf']=[1]*len(dfTR.index)

    #print("last t_n = ", Y.index[len(Y)-1], Y.loc[Y.index[len(Y)-1], ])
    #  classifier prediction column
    dfTR.loc[:, 'p_n'] = [0] * dfTR.index.size

    # Each prediction is for one day in the future

    last_i=dfTR.index[0]
    for i in dfTR.index[0:len(dfTR.index)-(nday-1)]:

        train_st2=train_st
        ndaydt = dt.timedelta(days=trainndays)
        tmp_st=i - ndaydt
        if tmp_st > train_st:
            train_st2=tmp_st

        dfTR.loc[i, 'train_st'] = train_st2

        X2 = X.loc[train_st2:last_i]
        Y2 = Y.loc[train_st2:last_i,'t_n']

        if i == dfTR.index[0]: # first time do not have a trained model
            mp = -1
            clf.fit(X2.as_matrix(), Y2.values.ravel())
            #print(last_i,'train')
            train_et=last_i

        else:
            #print(i)
            mp = clf.predict([X.loc[i]])
            dfTR.loc[i, 'p_n'] = mp
            #print(i,'predict')
            #print("")


        if dfTR.loc[i,'tf']==1:
            #if pd.notnull(Y2.loc[llast_i, 't_np1']):
            train_et=last_i
            try:
                clf.fit(X2.as_matrix(), Y2.values.ravel())
                #print(last_i,'train')
            except ValueError:
                e = sys.exc_info()[0]
                print("Exception occurred, ValueError")
                print(i,last_i)
                #print(Y2)
                #print(X2)
                pass

        dfTR.loc[i, 'train_et'] = train_et

        if v==1:
            if i.year != year:
                print(i.strftime('%Y-%m-%d'))

        elif v==2:
            if i.year != year or i.month != month:
                print(i.strftime('%Y-%m-%d'))


            # next loop variables
        year = i.year
        month = i.month
        ## note, have to go back two days so that future truth (+1 day) data does not pollute the training and prediction for current day
        last_i=i

    sys.stdout.write('\n')

    dfTR['t']= dfTR['t_n'].shift(nday)
    dfTR['p']= dfTR['p_n'].shift(nday)
    dfTR['p_1'] = dfTR['p_n'].shift(nday-1)


    #print ("Training month", i.strftime('%Y-%m-%d'), "train_st = ",train_st, 'train_st2 = ',train_st2)
    return dfTR, clf

from sklearn.preprocessing import Imputer



#  def marketCycle(df,mcpricevariable,mcdp,mcup,mudLogic)
#  Market Cycle. Returns two Data Frames
#    inputs:  df :=  data frame with market returns,
#                    indexed by date in Y-M-D format
#    returns: dfmc := market cycle (e.g., bull bear) detailed DataFrame with additional columns describing the
#                        mcprice_varible :=  for S&P500 this will be "Close"
#                        mkt             :=  1 up market, -1 down market
#                        mchlm           :=  market cycle high/low marker  ... 1 for high mark, 0 for low mark
#                        newhlm          :=  market new high or new low marker, 1 or 0, respectively
#                        sdm             :=  market switch detection point
#                        mcupm           :=  market cycle up marker, 1 for up market cycle and 0 for down market, based on sdm
#                        mcnr            :=  market cycle normalized return
#                        mucdown         :=  Down percent from previous high point, e.g., when mucdown hits 20% starts a bear market
#                        mdcup           :=  Up percent from previous low point, e.g., when mdcup higs 21% start bull market
#                        dfhlm           :=  Days from high low mark
#                        muchp           :=  Market up high percent based on last sdm price
#                        mdclp           :=  Market down cycle low percent based on last sdm price
#
#                     according to diagram below
#             dfmcsummary := market cycle (e.g., bull bear) summary dataframe
#                     columns := [bb,startDate,endDate,startPrice,endPrice,BullHigh,BullHighDate,BearLow,BearLowDate]
#
#    example:
#           dfbb = bullBear(df)
#
# Dexcription
#
# t(index)  0  1  2  3  4  5  6  7  8  9 10  11 12 13 14  15
#          ---bull----->|<-bear->|<--bull--->|<---bear-->|<--bull ---
#    S&P                                     .                   .
#                       .                 .     .             .
#                    .     .           .         sdm       sdm
#                 .         sdm      sdm               .   .
#              .              .    .
#           .                    .
# Example
#  description: "sdm" points are the points where the change in market
#       is detected ("switch detected mark"), corresponding bull down
#       20% from it's high or bear up 21% from its low. Retroactively
#       we can know the bull high or bear low and corresponding
#
# Bull      1  1  1  1  1  0  0  0  1  1  1  1  0   0   0   1
#
# Bear      0  0  0  0  0  1  1  1  0  0  0  0  1   1   1   0
#
#
# muchm      0  0  0  0  1  0  0  0  0  0  1  0  0   0   0   0
#  bear high marker
#
# mdclm      0  0  0  0  0  0  1  0  0  0  0  0  0   1   0   0
#  bear low marker
#
#
# mcnr    :=  market cycle normalized return  returns.
#             divide by price at the beginning of each market cycle
#
# Definitions
# Bull Market  := Rising market. Measured from the lowest close
#     after the market has fallen 20% or more to the next high
#
# Bear Market  := Declining market. Is defined as the market
#     closing 20% down from it's previous high. It's duration
#     is the period from the previous high to the lowest close after
#     it has fallen 20% or more.
#
# Bear market begins when down 20% from Bull High
# Bull market begins when up 21% from bear low

#bullbear will receives data frame with market returns
# applies the bullBearLogic and returns Bull Bear Data frame
# dfbb := [date, S&P, bull, bear, switch, normBBRet ]

def marketCycle(df,initMarket,mcpricevariable,mcdp,mcup,mudLogic):

    #initialize state variables
    initialMarket=initMarket

    n = 0


    if initialMarket == 1:
        muc = 1
        mdc = 0
        mcupm = 1

    elif initialMarket == -1:
        muc = 0
        mdc = 1
        mcupm = 0


    mkt = initialMarket


    # initialize data frames
    #  dfmc details
    #  dfmc summary

    dfmc = df.copy()
    dfmc = dfmc.reindex(columns=[mcpricevariable, 'mkt','mchlm', 'newmhlm', 'sdm', 'mcupm', 'mcnr','mucdown','mdcup','mcudthr'])
    dfmc.loc[df.index[0],['mkt','mchlm','sdm']] = [mkt,0,0]

    # mchlm := market cycle high low marker
    # newmlm := new market low marker
    # newmhm := new market high marker

    index=dfmc.index[0]
    #print("index = ",index)

    dfmcsummary = pd.DataFrame({'mkt': [], 'startTime': [], 'endTime': [], 'startPrice': [], 'endPrice': [], 'mcnr': []}, index=[])

    # loop through each day of df (data frame with market S&P)


    # Initialize Variables


    mdclp = 0
    mdclowtime = df.index[0]
    mdchigh = df.iloc[0,0]
    mdchightime=0
    muclow = float(df.iloc[0,0])
    muclowtime = 0
    muchp=0
    muchightime = 0
    switch = 0
    lswp = float(df.iloc[0,0])


    h1Price = dfmc.loc[dfmc.index[0],mcpricevariable]
    h2Price = h1Price
    mdclow = float(dfmc.loc[dfmc.index[0],mcpricevariable])
    muchigh = float(dfmc.loc[dfmc.index[0],mcpricevariable])
    mucdown2=0
    mdcup2=0
    dfhlm = -1

    newhlm=0

    #lastEndTime=df.index[0]
    #lastEndPrice=df.iloc[0,0]

    lastEndTime = dfmc.index[0]
    lastEndPrice = dfmc.loc[dfmc.index[0]]


    st = dfmc.index[0]
    et = dfmc.index[0]

    #print("initialMarket =", initialMarket, 'muc = ', muc)

    #dfmc.loc[dfmc.index[0], 'mchlm'] = 1
    #print(dfmc.index[0],dfmc.loc[dfmc.index[0], 'mchlm'])

    lastMarket = initialMarket
    first_switch=0
    for i in dfmc.index:
        date = df.ix[i].name
        date = i
        price = df.ix[i, 0]



        (n, mcupm, mcudthr,newmhlm, lswp, muc,  mucdown, mucdown2, muclow, muclowtime, muchp, muchigh, muchightime, mdc, mdcup, mdcup2, mdclp, mdclow, mdclowtime, mdchigh,
                    mdchightime, switch, mkt, st, et, sp, ep) =  \
                     marketCycleLogic(price,h1Price,h2Price, date, n, mcupm, lswp, mcdp,mcup, mudLogic,
                    muc, mucdown2, muclow, muclowtime, muchp, muchigh, muchightime, mdc, mdcup2, mdclp, mdclow,
                    mdclowtime, mdchigh, mdchightime)




        #print('i = ',i , 'v =', price, "mcupm = ",mcupm, 'mkt = ',mkt, 'muchigh =',muchigh,'mdclow =',mdclow, 'st =',st,'et',et,'sp',sp,'ep',ep )
        #print('  mucdown =',mucdown, 'mdcup =', mdcup )

        if newmhlm == 1:
            dfhlm = -1
        dfhlm += 1


        #if first_switch == 0:
        #    if mdc ==1:
        #        muchigh=mdclow
        #        muchightime=mdclowtime
        #    elif muc ==1:
        #        mdclow=muchigh
        #        mdclowtime=muchightime



        dfmc.loc[i,'newmhlm']=newmhlm
        dfmc.loc[i, 'mcudthr'] = mcudthr
        dfmc.loc[i,'muchp']=muchp
        dfmc.loc[i,'mdclp']=mdclp
        dfmc.loc[i,'mucdown']= mucdown
        dfmc.loc[i,'mdcup']=mdcup
        dfmc.loc[i, 'mcupm'] = mcupm
        if dfhlm >= 100:
            dfmc.loc[i, 'dfhlm'] = 10
        else:
            dfmc.loc[i, 'dfhlm'] = dfhlm / 10

        # fill in dfmcsummary info
        if switch == 1:

            d = {'mkt': [lastMarket],'startTime': [st], 'endTime': [et], 'startPrice': [sp], 'endPrice': [ep] }
            lastMarket = -1*lastMarket
            #print('   **** SWITCH ***', d)
            dftmp = pd.DataFrame(d, index=[st])
            lastEndTime = et
            lastEndPrice = ep
            dfmcsummary = dfmcsummary.append(dftmp)
            dfmc.loc[st, 'mchlm'] = 1
            dfmc.loc[et, 'mchlm'] = 1

        # fill in dfmc with
        #    sdm   - switch detection marker
        #    mchlm - market cycle high low mark
        dfmc.loc[i,'sdm'] = switch


        if pd.isnull(dfmc.loc[i,'mchlm']):
            dfmc.loc[i, 'mchlm']=0


        h2Price = h1Price
        h1Price = dfmc.loc[i,mcpricevariable]


    #print(df mcsummary)

    # After the loop is complete, in most cases there will not be a switch detected at the end time,
    #   thus the tail end of the market will not be represented in the summary
    #   so, update dfmcsummary with the latest data ... include startTime, startPrice, endTime, endPrice
    if switch == 0:
        if muc == 1:
            mkt = 1
        else:
            mkt = -1
        d = {'mkt': [mkt],'startTime': [lastEndTime], 'endTime': [date], 'startPrice': [lastEndPrice], 'endPrice': [ dfmc.loc[dfmc.index[dfmc.index.size-1], mcpricevariable] ] }
        dftmp = pd.DataFrame(d, index = [lastEndTime])

        dfmcsummary = dfmcsummary.append(dftmp)

    # At this point the only dfmc columns with data are
    #   S&P and sdm (switch detection point)
    #   from these all other entries can be determined
    #   fill in the remainder
    if initialMarket == 1:
        lastMkt = 1
    elif initialMarket == -1:
        lastMkt = -1
    #dfmc.loc[dfmc.index[0], ['mkt']] = lastMkt
    mcStartPrice = float(dfmc.loc[dfmc.index[0], mcpricevariable])

    lastMkt = initialMarket
    for i in dfmc.index:
        dfmc.loc[i,'mkt'] = lastMkt

        mcnr=float(dfmc.loc[i, mcpricevariable]) / mcStartPrice - 1
        dfmc.loc[i,['mcnr']] = mcnr
        #
        #print(i, lastMkt,dfmc.loc[i,'mchlm'] )
        if dfmc.loc[i,'mchlm'] == 1 and lastMkt == 1:
            #     sprint(i)
            if i != dfmc.index[0]:
                lastMkt = -1
            mcStartPrice = float(dfmc.loc[i,mcpricevariable])
            #print(lastMkt)
        elif dfmc.loc[i,'mchlm'] == 1 and lastMkt == -1:
            #print(i)
            if i != dfmc.index[0]:
                lastMkt = 1
            mcStartPrice = float(dfmc.loc[i,mcpricevariable])
            #print(lastMkt)

    # Add normalized Returns to dfmcsummary
    for i in dfmcsummary.index:
        dfmcsummary.loc[i, ['mcnr']] = dfmc.loc[dfmcsummary.loc[i,'endTime'], 'mcnr']

    return (dfmc, dfmcsummary)


#################################################################
#  Market Cycle Logic
#

def marketCycleLogic(price,h1Price,h2Price,date,n,mcupm,lswp,mcdp,mcup,mudLogic,muc,mucdown2,muclow,muclowtime,muchp,muchigh,muchightime,mdc,mdcup2,mdclp,mdclow,mdclowtime,mdchigh,mdchightime):
    v = float(price)
    t = date
    switch = 0
    if muc ==1:
        mkt =1
    else:
        mkt =-1
    st=pd.NaT
    et=pd.NaT
    sp = float('nan')
    ep = float('nan')

    # mcupm := market up indicator ("buy")
    # mdc := market down cycle (1 or 0)
    # muc := market up cycle (1 or 0)
    # mcudp : market cycle up/down percent
    # mdclow := market down cycle low
    # mdchigh := high measured from market down low while in down market
    # muchp := market up percentage from last sdm switch point
    # mdclp := market down percentage from last sdm switch point

    newmhlm = 0

    mucdown=0
    mdcup=0
    if muc == 1:
        mdcup = 0
        mdcup2 = 0
        mucdown = (muchigh - v) / muchigh
        muchp = v/lswp - 1  # market cycle up percentage from last low point
        mcudthr = muchigh*(1-mcdp)

    if mdc == 1:
        mucdown = 0
        mucdown2 = 0
        mdcup =  (v - mdclow) / mdclow
        mdclp = v/lswp -1   # market cycle down percentage, from last high point
        mcudthr = mdclow*(1+mcup)

#    if (n==0 ) and (muc == 1):
#        mdclow = v
#        mdclowtime = t
#    if (n ==0) and (mdc == 1):
#        muchigh = v
#        muchightime = t

    if (n==0 ):
        mdclow = v
        mdclowtime = t
        muchigh = v
        muchightime = t




    n += 1

    #print('  mdc = ',mdc,'mdclow = ',mdclow , ' muc =', muc,'muchigh =', muchigh)

    if (muc == 1) and v > muchigh:
        newmhlm = 1
        mucdown = 0
        mucdown2 = 0
        muchigh = v
        mcudthr = muchigh*(1-mcdp)
        muchightime = t
        muclow = v         # reset the bull low to the bull high
        muclowtime = t
        # print("    ",t, "up", "bullhigh = ",bullhigh, "bulllow =", bulllow, "adjclose =", v, "bullup =", bullup)

    elif (muc == 1) and v < muclow:
        mucdown2 =  (muchigh - v) / muchigh
        mucdown = mucdown2
        mcudthr = muchigh * (1 - mcdp)
        if v < muclow:   # this if is not needed
            muclow = v
            muclowtime = t
            # switch from up -> down
        if mudLogic(1,mucdown,mcdp,v,mdclow,h1Price,h2Price):
            switch = 1
            mcudthr = mdclow * (1 + mcup)
            newmhlm = 1
            muc = 0
            mdc = 1
            mdchigh = v
            mkt = 1
            sp = mdclow
            ep = muchigh
            st = mdclowtime
            et = muchightime
            lswp = v
            muchp = 0
            mcupm = 0
            #print("switch = ", 1, "muc to mdc", "mdclow = ", mdclow, "muchigh= ", muclow, ", price = ", v)
            mdclow = v
            mcudthr = mdclow * (1 + mcup)
            mdclowtime = t
    elif (mdc == 1) and (v < mdclow):
        newmhlm = 1
        mdcup=0
        mdcup2 = 0
        mdclow = v
        mcudthr=mdclow*(1+mcup)
        mkt = -1
        mdclowtime = t
        mdchigh = v
        mdchightime = t

    elif (mdc == 1) and (v > mdclow):   # the second part of this v > mdclow is not needed
        mcudthr = mdclow * (1 + mcup)
        if v > mdchigh:    # this could be moved up to the elif
            mdcup2 = (v - mdclow) / mdclow
            mdcup = mdcup2
            mdchigh = v
            mdchightime = t
            rbrh=mdchigh
            rbrht=mdchightime
        #### switch from down -> up ###

        if mudLogic(-1,mdcup,mcup,v,mdclow,h1Price,h2Price) :
            # print(" bulllow = ", bulllow)
            switch = 1

            newmhlm = 1
            muc = 1
            mdc = 0
            mdchigh = v
            mdchightime = t
            mkt = 1
            sp = muchigh
            ep = mdclow
            st = muchightime
            et = mdclowtime
            lswp = v
            mdclp=0
            mcupm = 1
            #print("switch = ", 1, "mdc to muc", "muchigh = ", muchigh, "mdclow = ", mdclow, ", price = ", v)
            muchigh = v
            mcudthr = muchigh * (1 - mcdp)
            muchightime = t

    return n, mcupm, mcudthr, newmhlm, lswp, muc, mucdown, mucdown2, muclow, muclowtime, muchp, muchigh, muchightime, mdc, mdcup, mdcup2, mdclp, mdclow, mdclowtime, mdchigh, mdchightime, switch, mkt, st, et, sp, ep


def mudLogic1(mkt,mud,mudp,price,muswdp,h1Price,h2Price):

    TF = False

    if mud > mudp:
        TF = True

    return TF


def mudLogic2(mkt,mud,mudp,price,muswdp,h1Price,h2Price):
    # muswdp := market up switch detection price (price when up market to down markdt is detected)

    TF = False

    if mkt == 1:
        if mud > mudp:
            TF = True
    if mkt == -1:
        if mud > mudp or price > muswdp:
            TF = True

    return TF

def mudLogic3(mkt,mud,mudp,price,muswdp,h1Price,h2Price):

    TF = False

    if mkt == 1:
        if mud > mudp:
            TF = True
    if mkt == -1:
        if mud > mudp and h1Price > h2Price:
            TF = True

    return TF

def mudLogic4(mkt,mud,mudp,price,muswdp,h1Price,h2Price):
    # muswdp := market up switch detection price (price when up market to down markdt is detected)

    TF = False

    if mkt == 1:
        if mud > mudp:
            TF = True
    if mkt == -1:
        if (mud > mudp or price > muswdp) and h1Price > h2Price:
            TF = True

    return TF

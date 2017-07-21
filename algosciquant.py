# Quant Functions

import pandas as pd
import pandas_datareader as web
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import rcParams


import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier



def getStockData(ticker,startDate,endDate,v=0,source='google'):
    import sys
    sDate= dt.datetime.strptime(startDate, "%Y-%m-%d")
    eDate = dt.datetime.strptime(endDate, "%Y-%m-%d")
    try:
        dfs = web.DataReader(ticker, source, sDate, eDate)
        dfs.sort_index(ascending=True, inplace=True)
        dfs.rename(columns={'Close': 'adj_close_price', 'Open': 'open_price', 'High': 'high_price', 'Low': 'low_price',
                            'Volume': 'volume'}, inplace=True)


    except:
        print('Exception: check your ticker, startDate and endDate and make sure there is data available.')
        print('ticker = ', ticker)
        print('startDate = ', startDate)
        print('endDate =', endDate)
        e = sys.exc_info()[0]
        print(e)

    return dfs


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



def getSpData(startDate,endDate,v = 0):

    dfsp=getStockData('^GSPC',startDate,endDate,v=0,source='yahoo')
    del dfsp['adj_close_price']
    if v > 0:
        print(dfsp.head(5))
    return dfsp


def getIntrinioSPData(startDate,endDate,api_username,api_password,items=[],v=1):
    #https://api.intrinio.com/historical_data?ticker=$SPX&item=open_price&start_date=2007-01-01&end_date=2017-02-17
    ticker = "$SPX"

    items2 = [ 'close_price',  'high_price',  'low_price',  'open_price',  'volume']

    if items:
        items2=items

    dfh = getIntrinioStockData(ticker, startDate, endDate, api_username, api_password, items2, v=v)

    return dfh


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

# Nday Truth

def plot_stock(df,start_date,end_date,plot_variables=['close_price'],labels=['close_price'],figsize=[12,4],loc='upper left',save_fig='',ncol=1):
    rcParams['figure.figsize'] = figsize
    fig = plt.figure()
    f = fig.add_subplot(111)
    for k in range(0,len(plot_variables)):
        f.plot(df.loc[start_date:end_date,plot_variables[k]],label=labels[k])
    plt.grid(True)
    f.legend(loc=loc,ncol=ncol)
    if save_fig:
        fig.savefig('sp_'+str(start_date.year)+'_'+str(end_date.year)+str(end_date.month)+str(end_date.day)+'.png')

def subplot_stock(fig,df,nsubplot,start_date, end_date,plot_variables=['close_price'],labels=['close_price'],figsize=[12,6],loc='upper left',save_fig='',ncol=1):
    rcParams['figure.figsize'] = figsize
    f = fig.add_subplot(nsubplot)
    for k in range(0,len(plot_variables)):
        f.plot(df.loc[start_date:end_date,plot_variables[k]],label=labels[k])
    plt.grid(True)
    f.legend(loc=loc,ncol=ncol)
    if save_fig:
        fig.savefig('sp_'+str(start_date.year)+'_'+str(end_date.year)+str(end_date.month)+str(end_date.day)+'.png')
    return

def normalized_plot_from_files(fig,sdate,edate,files,lnames,plot_variable='close_price',nsubplot=111,grid=True,loc='upper left',ncol=1,sfig=0,sfigname='fplot.png',figsize=[12,6]):
    k=0
    lnlist=[]

    for k in range(0,len(lnames)):
        if k==0:
            df= pd.read_csv(files[k],index_col=0,parse_dates=True)
            dftmp = pd.DataFrame(df.loc[sdate:edate, [plot_variable]])
            dftn_i0 = dftmp.index[0]
            dftn=pd.DataFrame(dftmp.loc[sdate:edate,lnames]/dftmp.loc[dftn_i0,lnames])
            dftn_i0 = dftn.index[0]
            dftn.rename(columns={plot_variable:lnames[k]},inplace=True)
        else:
            df=pd.read_csv(files[k],index_col=0,parse_dates=True)
            dftn[lnames[k]]=df.loc[sdate:edate,plot_variable+'_SP']/df.loc[dftn_i0,plot_variable+'_SP']
        lnlist.append(lnames[k])
        k+=1
    lnames = lnlist


    f=fig.add_subplot(nsubplot)
    for k in lnames:
        f.plot(dftn[k],label=k)
    f.grid(grid)
    f.legend(loc=loc, ncol=ncol)

    if sfig == 1:
        plt.savefig(sfigname)

    return

# ndayTruth
#   Inputs
#    df - DataFrame of stock prices. Example dfsp S&P or security
#    nday - compute lables for nday's in the future
#   Return
#    dfTruth - compute labels nday in the future +1 nday future price > current day, else -1
def ndayTruth(df,nday,tvariable='adj_close_price'):
    dfTruth=pd.DataFrame(df[tvariable])
    dfTruth['t_n']=np.sign((dfTruth[tvariable].shift(-nday)-dfTruth[tvariable])-0.00001)
    return dfTruth


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
        return row['p_1'] if row['vol_y_' + vlty] > vlty_thr and row['close_pricer_ma' + ma] < 0 else row[mcvariable+'_1']

    def fmcv(row):
        return -1 if row['vol_y_' + vlty] > vlty_thr and row['close_pricer_ma' + ma] < 0 else row[mcvariable+'_1']

    def fpv(row):
        return row['p_1'] if row['vol_y_'+ vlty] > vlty_thr and row['close_pricer_ma'+ma] < 0 else 1

    def fv(row):
        return -1 if row['vol_y_'+ vlty] > vlty_thr and row['close_pricer_ma'+ma] < 0 else 1

    def fp(row):
        return row['p_1'] if row[mcvariable] < 1 else 1

    if mcvariable:
        dfTR2[mcvariable+'_1'] = dfTR[mcvariable]
        dfTR2[mcvariable+'v_1'] = dfTR2.apply(fmcv, axis=1)
        dfTR2[mcvariable + 'pv_1'] = dfTR2.apply(fmcpv, axis=1)
        dfTR2[mcvariable + 'p_1'] = dfTR2.apply(fmcpv, axis=1)
    dfTR2['pv_1'] = dfTR2.apply(fpv, axis=1)
    dfTR2['v_1'] = dfTR2.apply(fv, axis=1)
    dfTR2['pv'] = dfTR2['pv_1'].shift(1)
    dfTR2['v'] = dfTR2['v_1'].shift(1)
    dfTR2[mcvariable] = dfTR2[mcvariable +'_1'].shift(1)
    dfTR2[mcvariable+'v'] = dfTR2[mcvariable+'v_1'].shift(1)
    dfTR2[mcvariable + 'pv'] = dfTR2[mcvariable + 'pv_1'].shift(1)
    dfTR2[mcvariable + 'p'] = dfTR2[mcvariable + 'p_1'].shift(1)


    return dfTR2

#################################################################
#  Confustion Matrix
#    Market Pred Confusion Matrix
#
def mktPredConfusionMatrix(df,truth,classified):


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

    dfCMA=pd.DataFrame({'Predicted MktDown':[tp,fp],'Predicted MktUp':[fn,tn],'Totals':[pos, neg]}, index=['actual MktDown','actual MktUp'])
    dfCMR=pd.DataFrame({'Predicted MktDown':[tpr,fpr],'Predicted MktUp':[fnr,tnr],'Totals': [pos, neg]}, index=['actual MktDown','actual MktUp'])

    return samplesize, errors, correct, er, dfCMA, dfCMR

def mlSpFeatures(dfsp,dfmc,mcvariable,dataStartDate,test_et):


    dfML = pd.DataFrame(index=dfsp.index)

    dfML['close_pricer'] = dfsp['close_price'] / dfsp['close_price'].shift(1) - 1
    dfML['volumer'] = dfsp['volume'] / dfsp['volume'].shift(1) - 1

    # S&P High and Low Relative to Open
    dfML['close_price'] = dfsp.loc[dataStartDate:test_et, 'close_price']
    dfML['volume'] = dfsp.loc[dataStartDate:test_et, 'volume']
    dfML['high_price_ropen'] = dfsp.loc[dataStartDate:test_et, 'high_price'] / dfsp['open_price'] - 1
    dfML['low_price_ropen'] = dfsp.loc[dataStartDate:test_et, 'low_price'] / dfsp['open_price'] - 1

    f2p = [2, 5, 10, 20, 30, 60, 90, 120]
    # Historya nd MA Features
    mlHmaFeatures = ['close_pricer', 'volumer']
    for mlHmaFeature in mlHmaFeatures:
        x = mlHmaFeature
        # History, history = i
        for i in range(1, f2p[0] + 1):
            dfML[x + '_h' + str(i)] = dfML[x].shift(i)

        # Moving averages, period=i
        for i in f2p[1:len(f2p) + 1]:
            period = i
            dfML[x + '_ma' + str(i)] = (dfML[x].rolling(center=False, min_periods=period, window=period).sum() / period)

    # Price Volatility
    period = 10
    dfML['vol_y_10'] = np.sqrt(252) * dfML['close_pricer'].rolling(center=False, min_periods=period,
                                                                   window=period).std()
    period = 50
    dfML['vol_y_50'] = np.sqrt(252) * dfML['close_pricer'].rolling(center=False, min_periods=period,
                                                                   window=period).std()
    period = 120
    dfML['vol_y_120'] = np.sqrt(252) * dfML['close_pricer'].rolling(center=False, min_periods=period,
                                                                    window=period).std()

    # Market Cycle ML Features
    dfML['mc'+mcvariable] = dfmc.loc[dataStartDate:test_et, 'mcupm'].shift(1)
    dfML['mcupm'] = dfmc.loc[dataStartDate:test_et, 'mcupm'].shift(1)
    dfML['mcnr'] = dfmc.loc[dataStartDate:test_et, 'mcnr']
    dfML['mucdown'] = dfmc.loc[dataStartDate:test_et, 'mucdown']
    dfML['mdcup'] = dfmc.loc[dataStartDate:test_et, 'mdcup']

    return dfML

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




def backTestSummary(dfTR,dfsp,price_variable,predictor,start_date,end_date):

    dft = tradeReturns(dfTR, price_variable)
    dft = strategyTrade(dft, price_variable, price_variable + '_R', predictor)
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
        dftsummary.loc[year, 'start_price'] = dft.ix[six, price_variable]
        dftsummary.loc[year, 'end_date'] = dft.index[eix]
        dftsummary.loc[year, 'end_' + price_variable] = dft.ix[eix, price_variable]

        if year == startyr:
            dftsummary.loc[year, 'start_' + price_variable + '_SP'] = dftsummary.loc[year,'start_price' ]   #dft.ix[six, price_variable]
        else:
            dftsummary.loc[year, 'start_' + price_variable + '_SP'] = dftsummary.ix[
                last_year, 'end_' + price_variable + '_SP']
        dftsummary.loc[year, 'end_' + price_variable + '_SP'] = dft.ix[eix, price_variable + '_SP']
        dftsummary.loc[year, 'return'] = dftsummary.loc[year, 'end_' + price_variable] / dftsummary.loc[
            year, 'start_price'] - 1


        dftsummary.loc[year, 'return_SP'] = dftsummary.loc[year, 'end_' + price_variable + '_SP'] / float(dftsummary.loc[
            year, 'start_' + price_variable + '_SP']) - 1
        last_year = year


    n = (end_strategy_trade - start_strategy_trade).days / (365)
    Rc = dftsummary.loc[endyr, 'end_' + price_variable] / dftsummary.loc[startyr, 'start_price'] - 1
    Rc_strat = dftsummary.loc[endyr, 'end_' + price_variable + '_SP'] / dftsummary.loc[
        startyr, 'start_' + price_variable + '_SP'] - 1

    Ra = ((Rc + 1) ** (1 / n)) - 1
    Ra_strat = ((Rc_strat + 1) ** (1 / n)) - 1


    d={ 'nyear':n,  'Ra': Ra,'Ra_strat': Ra_strat,'Rc': Rc,'Rc_strat': Rc_strat}
    dfreturns=pd.DataFrame(d,index=[end_date])


    return dftsummary, dfreturns

def mktClfTrainTest(X, Y, nday, train_st, test_st, test_et, model='DT', trainndays = -1,mc=0,dftflag=pd.DataFrame(),v=1):


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
    dfTR['t_1'] = dfTR['t_n'].shift(nday - 1)
    dfTR['p_1'] = dfTR['p_n'].shift(nday-1)


    dfTR.loc[test_et,'t_1']='NaN'



    #print ("Training month", i.strftime('%Y-%m-%d'), "train_st = ",train_st, 'train_st2 = ',train_st2)
    return dfTR, clf

def compute_market_cycle(dfsp,dataStartDate,end_date,mcdown_p=20,mcup_p=21,initMarket=1,save=1,v=1):
    mcvariable=str(mcdown_p)+str(mcup_p)
    mcdp=mcdown_p/100
    mcup=mcup_p/100
    print('mcvariable =',mcvariable,'\mcdown =',mcdown_p,'\mcup =',mcup_p)
    mudLogic=mudLogic1
    (dfmc, dfmcsummary)=marketCycle(dfsp.loc[dataStartDate:],initMarket,'close_price',mcdp,mcup,mudLogic)
    if v==1:
        print(dfmcsummary.tail(10))
    if save==1:
        # save market cycle
        save_dfmc_filename='sp500_dfmc'+str(mcdown_p)+str(mcup_p)+'_'+str(dataStartDate.year)+'_'+str(end_date.year)+'-'+str(end_date.month)+'-'+str(end_date.day)+'.csv'
        save_dfmcs_filename='sp500_dfmcs'+str(mcdown_p)+str(mcup_p)+'_'+str(dataStartDate.year)+'_'+str(end_date.year)+'-'+str(end_date.month)+'-'+str(end_date.day)+'.csv'
        print('filename =',save_dfmc_filename)
        dfmc.to_csv(save_dfmc_filename)
        dfmcsummary.to_csv(save_dfmcs_filename)
    return dfmc, dfmcsummary

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

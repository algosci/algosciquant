# Files Overview


## Algo Sci ndex.md file
index.md  
  
   
## Python Quant Functions   

algosciquant.py  


# Stock Prediction

doStockPred
  
doUpdateSPData     
   
## Get Data Scripts   
doUpdateStockDataIntrinio  - Updates data stock data, Intrinio. Calls getStock DataIntrinio.py  

appendTSFiles.py  - appends two time-series. called by doUpdateStockData ... to append a small update with a historical stock data file.

getSP500QDL.py  

getSP500Yahoo.py  

getSPDataIntrino.py  

getSPDataYahoo.py  

getStockDataIntrino.py  

getStockDataYahoo.py  


## Market Cycle

marketCycle.py - market cycle python functions

spMarketCycle.py - python script, callable by the shell

SPred.ipynb - stock Prediction, Jupyter notebook

doSpMarketCycle - shell script, calls spMarketCcyle
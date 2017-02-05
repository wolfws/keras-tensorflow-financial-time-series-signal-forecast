import numpy as np
import os
import pywt
import pandas as pd
import mysql.connector
np.set_printoptions(suppress=True)
from statsmodels.tsa.api import VAR
from statsmodels.robust import mad

riskfree = 0 
products = ['EQUITY','FX','ETF','INDEX','UST','COMMODITIES']
wavelet = 'sym4'
padding = 'smooth'
dlevel = 3
thresh=10

def wave(wData):
    newWL = wData.copy() 
    for i in range(0,wData.shape[1]):
        nblck = wData[wData.columns[i]].copy() #!
        noisy_coefs = pywt.wavedec(nblck, wavelet, level=dlevel, mode=padding)
        sigma = mad(noisy_coefs[-1]).copy()
        uthresh = sigma*thresh
        denoised = noisy_coefs[:]
        denoised[1:] = (pywt.threshold(i, uthresh, 'soft') for i in denoised[1:]) # smoothing
        newWL[wData.columns[i]] = pywt.waverec(denoised, wavelet, mode=padding).copy()
    return newWL

def loadForecastFilesToDb():
	pass

def loadSTDFileToDb():
	pass
	
def getProductPaths():
	pass
	
def getTS():
	pass

def saveStd():
	pass
	
def saveStats():
	pass

def replaceOutlier(TS):
    for n in list(TS.columns):
        length = len(TS[n])
        for i in range(0, length, 100):
            p_mean = TS[n][i:i+100].mean()
            l_bound = p_mean * 0.1 
            u_bound = p_mean * 3 
            TS[n][i:i+100][TS[n][i:i+100] > u_bound] = 0
            TS[n][i:i+100][TS[n][i:i+100] < l_bound] = 0
        TS[n].replace(to_replace=0, method='bfill', inplace = True)
    return TS

	# Vector Auto-regression Multivariate Time Series Forecast Function
def var(ds): #numpy array input
    lag = 10;days = 5
    dslog = np.log(ds)
    df = np.nan_to_num(np.diff(dslog, axis=0))
    model = VAR(df)
    results = model.fit(maxlags=lag, ic='aic')
    lag_order = results.k_ar 
    if lag_order < 5:
        lag_order = 10
    fc = results.forecast(df[-lag_order:], days)
    fcdenorm = np.exp(np.cumsum(fc,axis=0)+ dslog[-1:])
    fcdenorm = np.vstack((ds[-1:],fcdenorm)) 
    return np.round(fcdenorm[-5:],5),lag_order


if __name__ == '__main__':


    for product in products:

		# function to get product paths	

        for f in getProductPaths():
            try:
                ds = getTS(f)
                if len(ds) % 2 != 0:
                    ds.drop(ds.index[:1], inplace=True)
                tickers = list(ds.columns)
                X = (ds.shape[1]) + 10*(ds.shape[1])**2   #60 training window, days  minlag = columns + lags*columns**2
                if X % 2 != 0:
                    X = X + 1
                if X < 300:
                    X = 300
                total = X + 100
                ds = replaceOutlier(ds[-total:])
				ds = ds.astype('float32')
                days = 5 #days to forecast = total tickers in the groups. later we only keep the first 5 to forecast 5 periods forward
                remainder = total - X - days
                data = ds
                predictX = [data[i:X+i] for i in range(1,remainder-1)]
                predictY = [data[X+i:X+i+days] for i in range(1,remainder-1)]
                predPX_lst = []
				myi = 0;
                fc = []
				stats = []
				basetbl = pd.DataFrame(columns = ['date','day','ticker','fc','actual','lastpx'])
                for i in range(len(predictX)):
                    try:
                        tempPX_lst = []
						tempPct_lst = []   #need to find 1st date for predictX.shape[0]
                        start = X + i
                        end = X + i + days
                        actualprices = ds[start:end]
                        lastdayprices = ds[X +i - 1:X + i]
                        pX = wave(predictX[i])
                        pX = pX.values
                        fcast, lag = var(pX)
                        actualprices = ds[start:end]
						lastdayprices = ds[X +i - 1:X + i]
                        fcast = pd.DataFrame(fcast)
						fcast.columns = tickers
                        fcast['day']=fcast.index
                        fcast.index = actualprices.index
						fcast['date']=fcast.index
                        fcast = pd.melt(fcast, id_vars = ['date','day'])
						fcast.columns = ['date','day','ticker','fc']    
                        actualprices['date']= actualprices.index
                        actualprices = pd.melt(actualprices, id_vars=['date'])
						actualprices.columns = ['date','ticker','actual']
                        basetbl1 = pd.concat([fcast, actualprices['actual']], axis=1, join='inner')
                        basetbl1['lastpx']=0.00
                        for t in list(lastdayprices.columns):
                            px = lastdayprices[t].iloc[0]
                            basetbl1.loc[basetbl1['ticker']==t,'lastpx']=px
                        basetbl = basetbl.append(basetbl1, ignore_index = True)
                    except:
                        # handle errors here
                        myi +=1
                        continue

                if len(predictX)  - myi < 3:
                    # handle error here
                    continue
            
                d = basetbl
                d['std'] = 0.00
				d['std_perc'] = 0.00
                dz = [0, 1, 2, 3, 4]
                stdtbl = []
                for ticker in tickers:
                    for day in dz:
                        d1 = d[ (d['ticker']== ticker) & (d['day']==day )]
                        std1 = np.std(np.nan_to_num(d1['fc'].values) - np.nan_to_num(d1['actual'].values), axis = 0)
                        std1_pct = np.std(100*(np.nan_to_num(d1['fc'].values) - np.nan_to_num(d1['actual'].values) )/np.nan_to_num(d1['fc'].values), axis=0)
                        basetbl.loc[(basetbl['ticker']==ticker) & (basetbl['day']==day), 'std'] = std1
                        basetbl.loc[(basetbl['ticker']==ticker) & (basetbl['day']==day), 'std_perc'] = std1_pct
                        stdtbl.append((1,day,ticker,std1,std1_pct))

                stdtbl = pd.DataFrame(stdtbl)
				stdtbl.columns = ['modelid','day','ticker','std','std_perc']
				saveStd(stdtbl)
                
                g = basetbl
                # *** PERFORMANCE STATS
                sdrange = [0.25, 0.5, 1, 1.5, 2, 2.5]
                for ticker in tickers:
                    for day in dz:
                        for sd in sdrange:
                            g['trade_signal']= 0;g['pnl']=0
                            g['trade_signal'] = np.where( (g['fc'] - g['std']*sd) > g['lastpx'], 1, g['trade_signal'])
                            g['trade_signal'] = np.where( (g['fc'] + g['std']*sd) < g['lastpx'], -1, g['trade_signal'])	
                            g['pnl'] = np.where(g['trade_signal'] == 1, g['actual'] - g['lastpx'], g['pnl'])
                            g['pnl'] = np.where(g['trade_signal'] == -1, g['lastpx'] - g['actual'], g['pnl'])

                            d1 = g[ (g['ticker']== ticker) & (g['day']==day )]
                            pnl = d1['pnl'].sum()
							longs = len(d1[d1['trade_signal']== 1])
							shorts = len(d1[d1['trade_signal']== -1])
                            avgwin =d1[d1['pnl']> 0]['pnl'].mean()
							avgloss =d1[d1['pnl']< 0]['pnl'].mean()
                            winners =d1[d1['pnl']> 0]['pnl'].count()
							losers =d1[d1['pnl'] < 0]['pnl'].count()
                            z1 = np.std(np.diff(np.log(d1['actual'].values)))
                            stdev = (z1 / np.sqrt(d1['actual'].values.shape[0]) ) * np.sqrt(252)
							hpr_perc = pnl /  d1[:1]['actual'].iloc[0]
                            sharpe = (((1 + hpr_perc)**(1/(d1['actual'].values.shape[0]/252)) - 1) - riskfree) / stdev
                            ann_return = (1 + hpr_perc)** (1/(d1['actual'].values.shape[0]/252)) - 1
                            std1 = np.std(np.nan_to_num(d1['fc'].values) - np.nan_to_num(d1['actual'].values), axis = 0)
                            std1_pct = np.std(100*(np.nan_to_num(d1['fc'].values) - np.nan_to_num(d1['actual'].values) )/np.nan_to_num(d1['fc'].values), axis=0)
                            stats.append([1,ticker,day,sharpe,pnl,ann_return, hpr_perc, winners, losers,avgwin, avgloss,longs,shorts,std1,std1_pct,sd])

                s1 = pd.DataFrame(stats)
				s1.columns = ['m','t','p','shr','pnl','ret','hpr','win','lsr','avwin','avlsr','lngs','shrt','std','stdp','sd']
                s1.fillna(0, inplace=True)
                s1 = s1.round({'shr':1,'pnl':2,'ret':4,'hpr':4,'avwin':4,'avlsr':4,'std':4,'stdp':4})
                s1['p'] = s1['p'].apply(lambda x: x + 1)
                s1['sim']= len(predictX)
                s1['window'] = len(predictX[0])
				saveStats(s1)

            except:
				# handle errors here

    loadSTDFileToDb()                                    
    loadForecastFilesToDb()

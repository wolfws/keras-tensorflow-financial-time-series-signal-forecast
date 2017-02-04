import pywt
import mysql.connector
cmd = ['python', '-c', 'print(hash("abc"))']
call(cmd, env={'PYTHONHASHSEED': bytes(0)})
import numpy as np
import os
np.random.seed(1337) #fix random seed to get deterministic results
import pandas as pd
import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dense, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from statsmodels.robust import mad
from datetime import tzinfo, timedelta, datetime
from keras.optimizers import SGD, RMSprop, Adagrad
from subprocess import call
from statsmodels.robust import mad

np.set_printoptions(suppress=True)

wavelet = 'sym4'
padding = 'smooth'
dlevel = 3
thresh=10
mc = 100
riskfree = 0
X = 30
daysback=250 #training window, days

products = ['FX','ETF','EQUITY','INDEX','UST']


def wave(wData):
    newWL = wData.copy() 
    for i in range(0,wData.shape[1]):
        nblck = wData[wData.columns[i]].copy() 
        noisy_coefs = pywt.wavedec(nblck, wavelet, level=dlevel, mode=padding)
        sigma = mad(noisy_coefs[-1]).copy()
        uthresh = sigma*thresh
        denoised = noisy_coefs[:]
        denoised[1:] = (pywt.threshold(i, uthresh, 'soft') for i in denoised[1:]) # smoothing
        newWL[wData.columns[i]] = pywt.waverec(denoised, wavelet, mode=padding).copy()
    return newWL

class CompileModelClass():
    def __init__(self):
        self.loaded_models = {} 
   
    def LoadModel(self, colnum):
        if colnum in self.loaded_models: #model exists in memory
            return self.loaded_models.get(colnum)
        else:
            features = colnum  
            hidden = X * colnum   
            model = Sequential()
            model.add(LSTM(hidden, input_shape=(X, features)))
            model.add(Dropout(.2))
            model.add(Dense(features))
            model.add(Activation('linear')) 
            model.compile(loss='mse', optimizer='rmsprop')     
            self.loaded_models[colnum] = model
            return model

def loadForecastToDb():
  pass
  
def getDS():
  pass
  
def getList():
  pass
  
def dsSave():
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


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss')) 


if __name__ == '__main__':
    
    c = CompileModelClass()
    for product in products:
        # ....
        # loop through pool of portfolios and products....
        # ....
        
        for f in getList():
            try:
                ds = getDS()
                if len(ds) % 2 != 0:
                    ds.drop(ds.index[:1], inplace=True)
                ds = replaceOutlier(ds[-daysback:]) 
                ds = wave(ds)
                tickers = list(ds.columns);total = len(ds);
                days = ds.shape[1] #days to forecast = total tickers in the portfolio subset
                remainder = total - X - days
                # *** NORMALIZE ****
                d1 = ds.astype('float32')
                df = np.nan_to_num(np.diff(np.log(d1), axis=0))
                data = 1 / (1 + np.exp( (df - df.mean()) / df.std() )  )
                features = ds.shape[1]  #num of dimensions aka columns                
                model = c.LoadModel(ds.shape[1])
                # *** TRAIN & TEST DATA PREP ****
                baseX = np.asarray([data[i:X+i] for i in range(1,remainder-1)]) 
                baseY = np.asarray([data[X+i] for i in range(1,remainder-1)])
                startX = int(np.round(0.7*len(baseX),0))
                startY = int(np.round(0.7*len(baseY),0))
                trainX = baseX[:startX]
                trainY = baseY[:startY] # reshape input to be [samples, time steps, features]
                predictX = baseX[startX:]
                predictY = baseY[startY:]
                # *** END TRAIN & TEST DATA PREP ****
                fc = []
                stats = []
                basetbl = pd.DataFrame(columns = ['actual','date','fc','day','lastpx','ticker'])
                for counter in range(len(tickers)): #data.shape[1] to ensure that forecast length equals # of ticker dimensions
                    history = LossHistory()
                    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                    trainY = np.asarray([data[:,counter][X+i:X+i+data.shape[1]] for i in range(1,remainder-1)])
                    trainY = baseY[:startY] 
                    model.fit(trainX, trainY, nb_epoch=30,  batch_size= 10, validation_split=0.20, callbacks=[early_stopping,history],verbose=0)

                    predPX_lst = []
                    left = -1
                    right = 0
                    for i in range(predictX.shape[0]):
                        tempPX_lst = []
                        tempPct_lst = []  
                        start = startX + X + i
                        end = startX + X + i + days
                        actualprices = ds[start:end]
                        lastdayprices = ds[startX + X +i - 1:startX + X + i]   
                        pX = predictX[i]
                        pX = np.reshape(pX, (1, pX.shape[0], pX.shape[1]))
                        forecast =[]
                        tempPred1 = []
                        for iii in range(mc):  #average n-day mc loop  
                            predicted = model.predict(pX)
                            tempPred1.append(predicted)  
                        predicted = np.mean(tempPred1,axis=0) #Average predictions generated by mc    
                        predicted2 = np.log( (1 - predicted)/predicted ) *df.std() + df.mean()
                        fcast = np.repeat(np.asarray(lastdayprices.ix[:,counter][-1:]),features) *(1 + predicted2)
                        fcast = np.reshape(fcast, (1,1,fcast.shape[1]))
                        predPX_lst.append((tickers[counter], fcast, lastdayprices, actualprices))
                        s = np.reshape(np.transpose(fcast), (np.transpose(fcast).shape[0], 1))
                        z1 = pd.DataFrame(actualprices[tickers[counter]])
                        z1.columns = ['actual']
                        z1['date']= z1.index
                        z1.reset_index(drop=True, inplace=True)
                        z2 = pd.concat( [z1, pd.DataFrame(s)], axis=1)
                        z2['day']= z2.index   
                        z2 = pd.concat([z2, pd.DataFrame(np.repeat(lastdayprices[tickers[counter]].iloc[0], ds.shape[1]))], axis=1)
                        z2['ticker']= tickers[counter]
                        z2.columns = ['actual','date','fc','day','lastpx','ticker']
                        z2 = z2[:5]
                        basetbl = basetbl.append(z2, ignore_index = True)

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
                        stdtbl.append((2,day,ticker,std1,std1_pct))

                stdtbl = pd.DataFrame(stdtbl);stdtbl.columns = ['modelid','day','ticker','std','std_perc']
                stdtbl = stdtbl.round({'std':4,'std_perc':4})
                dsSave(stdtbl)

                # *** P&L CALCULATIONS ***
                g = basetbl
                # *** PERFORMANCE STATS
                sdrange = [0.25, 0.5, 1, 1.5, 2, 2.5] #sigma signal range
                for ticker in tickers:
                    for day in dz:
                        for sd in sdrange:
                            g['trade_signal']=0
                            g['pnl']=0
                            g['trade_signal'] = np.where( (g['fc'] - g['std']*sd) > g['lastpx'], 1, g['trade_signal'])
                            g['trade_signal'] = np.where( (g['fc'] + g['std']*sd) < g['lastpx'], -1, g['trade_signal'])	
                            g['pnl'] = np.where(g['trade_signal'] == 1, g['actual'] - g['lastpx'], g['pnl'])
                            g['pnl'] = np.where(g['trade_signal'] == -1, g['lastpx'] - g['actual'], g['pnl'])
                            
                            d1 = g[ (g['ticker']== ticker) & (g['day']==day )]
                            pnl = d1['pnl'].sum();longs = len(d1[d1['trade_signal']== 1])
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
                            # *** simply repeat std to export to database
                            std1 = np.std(np.nan_to_num(d1['fc'].values) - np.nan_to_num(d1['actual'].values), axis = 0)
                            std1_pct = np.std(100*(np.nan_to_num(d1['fc'].values) - np.nan_to_num(d1['actual'].values) )/np.nan_to_num(d1['fc'].values), axis=0)
                            stats.append([2,ticker,day,sharpe,pnl,ann_return, hpr_perc, winners, losers,avgwin, avgloss,longs,shorts,std1,std1_pct,sd])
                        
                s1 = pd.DataFrame(stats)
                s1.columns = ['m','t','p','shr','pnl','ret','hpr','win','lsr','avwin','avlsr','lngs','shrt','std','stdp','sd']
                s1.fillna(0, inplace=True)
                s1 = s1.round({'shr':1,'pnl':2,'ret':4,'hpr':4,'avwin':4,'avlsr':4,'std':4,'stdp':4})
                s1['p'] = s1['p'].apply(lambda x: x + 1)
                s1['sim']= predictX.shape[0]
                s1['window'] = X
                dsSave(s1)
            except:
                pass # handle errors here    
                               
    loadForecastToDb()

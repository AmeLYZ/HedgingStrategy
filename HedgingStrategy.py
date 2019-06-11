# -*- coding:utf-8 -*-
# Python 2.7.15

import numpy as np
import pandas as pd

import datetime
import random
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# define the begin/end date and how many shares are held
# trade day '2010-04-30' '2010-05-07' '2010-05-14' '2010-05-21' '2010-05-28'
DATE1 = '2010-04-16'
BGNDATE = '2010-11-15'
ENDDATE = '2010-12-17'
HOLDSHARE = 80  # 80 * 10000
# IF1012
AGREEMENT = u'IF1012'  # 2010-04-16 to 2010-12-17

# index contract multiplier
MULTIPLIER = 300
TRAN_RATE = 0.00015
"""
data1 BETA
data2 SECURITY NAME
data3 DAYLY PRICE
data4 INDEX
data5 FUTURE
"""
data1 = pd.read_csv(r'data\BETA_Dbeta.txt', sep='\t', encoding='utf-16')
data2 = pd.read_csv(r'data\TRD_Co.txt', sep='\t', encoding='utf-16')
data3 = pd.read_csv(r'data\TRD_Dalyr.txt', sep='\t', encoding='utf-16')
data4 = pd.read_csv(r'data\TRD_Index.txt', sep='\t', encoding='utf-16')
data5 = pd.read_csv(r'data\FFUT_FDT.txt', sep='\t', encoding='utf-16')

set_beta = set(data1['Stkcd'])  # len 1832
set_co = set(data2['Stkcd'])  # len 3839
set_price = set(data3['Stkcd'])  # len 2140


set_trddt_1 = set(data1[data1['Stkcd'].isin([2])]['Trddt'])  # len 482
set_trddt_2 = set(data1[data1['Stkcd'].isin([550])]['Trddt'])  # len 481
set_trddt_3 = set(data1[data1['Stkcd'].isin([2024])]['Trddt'])  # len 479

# all the stocks with valid data
STOCK_LIST = sorted(list(set_beta.intersection(set_co).intersection(set_price)))
TRADE_DATE_LIST = sorted(list(set_trddt_1.intersection(set_trddt_2).intersection(set_trddt_3)))  # len 470
TRADE_DATE = sorted(list(set(data1['Trddt'])))

# get the sample portfolio
PORTFOLIO_CODE_LIST = [1L, 2L, 63L, 550L, 989L, 2024L, 2041L, 2160L, 600050L,\
                  600310L, 600583L, 601958L, 601988L, 601991L, 601998L, 601999L]


# PORTFOLIO_CODE_LIST = sorted(random.sample(STOCK_LIST), 20))

'''
check valid values
len(set(beta['Stkcd']))
len(set(portfolio_info['Stkcd']))
len(set(closing_price['Stkcd']))
'''

# beta[beta['Stkcd']==601999].sort_values(by="Trddt")
# db.future_data[db.future_data['Trddt']==BGNDATE]['Deldt']



class Database(object):
    """ traditional Model"""
    def __init__(self, portfolio_code_list=PORTFOLIO_CODE_LIST):
        # get the beta of each stock
        beta = data1[data1['Stkcd'].isin(portfolio_code_list)].copy()

        # get the portfolio basic infomation
        portfolio_info = data2[data2['Stkcd'].isin(portfolio_code_list)][['Stkcd', 'Markettype', 'Stknme']].copy()
        portfolio_info = portfolio_info.dropna().sort_values('Stkcd').reset_index(drop=True)
        holding = pd.DataFrame([HOLDSHARE] * len(portfolio_code_list), columns=['Hldshr'])
        portfolio_info = pd.concat([portfolio_info, holding], axis=1)

        # get the closing price of each stock
        closing_price = data3[data3['Stkcd'].isin(portfolio_code_list)].copy()

        # get the data of CSI300 Index
        index_data = data4[data4['Indexcd'].isin(['000300'])].copy()

        # get the data of Stock Index Future
        future_data = data5[data5['Trdvar'].isin([u'沪深300指数期货'])].copy()

        self.portfolio_code_list = portfolio_code_list
        self.beta = beta.dropna().sort_values('Stkcd').reset_index(drop=True)
        self.portfolio_info = portfolio_info.dropna().sort_values('Stkcd').reset_index(drop=True)
        self.closing_price = closing_price.dropna().reset_index(drop=True)
        self.index_data = index_data.dropna().reset_index(drop=True)
        self.future_data = future_data.dropna().reset_index(drop=True)
        self.portfolio = pd.DataFrame()
        self.init_beta = None

    # init the portfolio (for display)
    def portfolio_init(self):
        # get the portfolio at time 0
        portfolio, _, _ = self.get_daily_info(BGNDATE, self.portfolio_code_list)

        self.portfolio = portfolio.dropna().reset_index(drop=True)


    # get the beta of portfolio at certain date
    def get_daily_info(self, date, prtfl_list):
        info = self.portfolio_info[['Stkcd', 'Markettype', 'Stknme']].copy()

        clspr = self.closing_price[(self.closing_price['Trddt'] == date) & (self.closing_price['Stkcd'].isin(prtfl_list))].copy()
        clspr = clspr[['Stkcd', 'Clsprc']].sort_values('Stkcd').reset_index(drop=True)

        hldshr = self.portfolio_info[self.portfolio_info['Stkcd'].isin(prtfl_list)].copy()
        hldshr = hldshr[['Stkcd', 'Hldshr']].sort_values('Stkcd').reset_index(drop=True)

        bt = self.beta[(self.beta['Trddt'] == date) & (self.beta['Stkcd'].isin(prtfl_list))].copy()
        bt = bt[['Stkcd', 'Betavals']].sort_values('Stkcd').reset_index(drop=True)

        info = pd.merge(info, clspr)
        info = pd.merge(info, hldshr)
        info = pd.merge(info, bt)

        info['Stk'] = info['Stkcd'].map(lambda x: '000000'+str(int(x))).map(lambda x: x[-6:])
        info['Markettype'] = info['Markettype'].map(lambda x: '.SZ' if x==4 else '.SH')
        info['Stk'] = info.apply(lambda x: x['Stk'] + x['Markettype'], axis=1)
        info['Vals'] = info.apply(lambda x: x['Clsprc'] * x['Hldshr'], axis=1)
        info['wght'] = info.apply(lambda x: x['Vals'] * x['Betavals'], axis=1)
        info = info[['Stk', 'Stknme', 'Clsprc', 'Hldshr', 'Betavals', 'Vals', 'wght', 'Stkcd']]

        prtfl_value = info['Vals'].sum()
        prtfl_bt = info['wght'].sum() / prtfl_value
        self.init_beta = prtfl_bt

        return info, prtfl_bt, prtfl_value



class TraditionalModel(object):
    """ traditional Model"""
    def __init__(self):
        self.h = 1.
        self.portfolio = list()
        self.n = 0.
        self.flt = float("-inf")

        self.stock_bgn = None
        self.stock_end = None

        self.index_bgn = None
        self.index_end = None


    def n_init(self):
        stock_value = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        self.n = round((stock_value * 10000 * self.h) / (self.index_bgn * MULTIPLIER), 3)
        # print '\nNeed ({})\t contracts to hedge the risk for filter (beta>{})'.format(self.n, self.flt)


    def hedge_init(self, db):
        portfolio, prtfl_bt, prtfl_value = db.get_daily_info(BGNDATE, db.portfolio_code_list)
        stock_bgn = db.portfolio[['Betavals', 'Vals', 'wght']].copy()

        portfolio, prtfl_bt, prtfl_value = db.get_daily_info(ENDDATE, db.portfolio_code_list)
        stock_end = portfolio[['Betavals', 'Vals', 'wght']].copy()

        index_bgn = db.future_data[(db.future_data['Trddt']==BGNDATE) & (db.future_data['Agmtcd']==AGREEMENT)]['Stprc'].values[0]
        index_end = db.future_data[(db.future_data['Trddt']==ENDDATE) & (db.future_data['Agmtcd']==AGREEMENT)]['Stprc'].values[0]

        self.stock_bgn = stock_bgn
        self.stock_end = stock_end
        self.index_bgn = index_bgn
        self.index_end = index_end


    def hedge(self):
        value_bgn = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        value_end = self.stock_end[self.stock_end['Betavals'] > self.flt]['Vals'].sum()

        spot_market = round(value_end - value_bgn, 3)
        index_market = round((self.index_bgn - self.index_end) * self.n * MULTIPLIER / 10000, 3)
        tran_cost = self.index_end * self.n * TRAN_RATE * MULTIPLIER / 10000
        result = round(spot_market + index_market - tran_cost, 3)

        print '\n| contracts\t| best h ratio\t| gain at spot\t| gain at index\t| result\t|'
        print '| {}\t| {}\t\t| {}\t| {}\t| {}\t|'.format(self.n, self.h, spot_market, index_market, result)


    def hedge_noprint(self):
        value_bgn = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        value_end = self.stock_end[self.stock_end['Betavals'] > self.flt]['Vals'].sum()

        spot_market = round(value_end - value_bgn, 3)
        index_market = round((self.index_bgn - self.index_end) * self.n * MULTIPLIER / 10000, 3)
        tran_cost = self.index_end * self.n * TRAN_RATE * MULTIPLIER / 10000
        result = round(spot_market + index_market - tran_cost, 3)

        profit_n = round(spot_market / value_bgn, 3)
        profit_h = round(result / value_bgn, 3)

        return profit_n, profit_h



class CAPMBetaModel(object):
    """ CAPM Beta Model"""
    def __init__(self):
        self.h = 0.
        self.portfolio = list()
        self.n = 0.
        self.flt = float("-inf")

        self.stock_bgn = None
        self.stock_end = None

        self.index_bgn = None
        self.index_end = None


    def n_init(self):
        h = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['wght'].sum() / self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        self.h = round(h, 3)

        stock_value = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        self.n = round((stock_value * 10000 * self.h) / (self.index_bgn * MULTIPLIER), 3)


    def hedge_init(self, db):
        portfolio, prtfl_bt, prtfl_value = db.get_daily_info(BGNDATE, db.portfolio_code_list)
        stock_bgn = db.portfolio[['Betavals', 'Vals', 'wght']].copy()

        portfolio, prtfl_bt, prtfl_value = db.get_daily_info(ENDDATE, db.portfolio_code_list)
        stock_end = portfolio[['Betavals', 'Vals', 'wght']].copy()

        index_bgn = db.future_data[(db.future_data['Trddt']==BGNDATE) & (db.future_data['Agmtcd']==AGREEMENT)]['Stprc'].values[0]
        index_end = db.future_data[(db.future_data['Trddt']==ENDDATE) & (db.future_data['Agmtcd']==AGREEMENT)]['Stprc'].values[0]

        self.stock_bgn = stock_bgn
        self.stock_end = stock_end
        self.index_bgn = index_bgn
        self.index_end = index_end


    def hedge(self):
        value_bgn = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        value_end = self.stock_end[self.stock_end['Betavals'] > self.flt]['Vals'].sum()

        spot_market = round(value_end - value_bgn, 3)
        index_market = round((self.index_bgn - self.index_end) * self.n * MULTIPLIER / 10000, 3)
        tran_cost = self.index_end * self.n * TRAN_RATE * MULTIPLIER / 10000
        result = round(spot_market + index_market - tran_cost, 3)

        print '\n| contracts\t| best h ratio\t| gain at spot\t| gain at index\t| result\t|'
        print '| {}\t| {}\t\t| {}\t| {}\t| {}\t|'.format(self.n,self.h, spot_market, index_market, result)


    def hedge_noprint(self):
        value_bgn = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        value_end = self.stock_end[self.stock_end['Betavals'] > self.flt]['Vals'].sum()

        spot_market = round(value_end - value_bgn, 3)
        index_market = round((self.index_bgn - self.index_end) * self.n * MULTIPLIER / 10000, 3)
        tran_cost = self.index_end * self.n * TRAN_RATE * MULTIPLIER / 10000
        result = round(spot_market + index_market - tran_cost, 3)

        profit_n = round(spot_market / value_bgn, 3)
        profit_h = round(result / value_bgn, 3)

        return profit_n, profit_h




class OLSModel(object):
    """ OLS Model"""
    def __init__(self, stock_bgn, stock_end, index_bgn, index_end, flt=float("-inf")):
        self.h = 0.
        self.portfolio = list()
        self.n = 0.
        self.flt = flt

        self.stock_bgn = stock_bgn
        self.stock_end = stock_end

        self.index_bgn = index_bgn
        self.index_end = index_end


    def n_init(self, db):
        date = np.array(TRADE_DATE)
        date = date[np.logical_and(date<'2010-12-17', date>'2010-04-16')]

        x = [[i, db.get_daily_info(i, db.portfolio_code_list)[2]] for i in date]
        S = pd.DataFrame(x)
        S.columns = ['Trddt', 'S']

        y = [[i, db.future_data[db.future_data['Trddt']==i]['Stprc'].values[3]] for i in date]
        F = pd.DataFrame(y)
        F.columns = ['Trddt', 'F']

        SF = pd.merge(S, F)

        h = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['wght'].sum() / self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        self.h = round(h, 3)

        stock_value = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        self.n = round((stock_value * 10000 * self.h) / (self.index_bgn * MULTIPLIER), 3)


    def hedge_init(self, db):
        portfolio, prtfl_bt, prtfl_value = db.get_daily_info(BGNDATE, db.portfolio_code_list)
        stock_bgn = db.portfolio[['Betavals', 'Vals', 'wght']].copy()

        portfolio, prtfl_bt, prtfl_value = db.get_daily_info(ENDDATE, db.portfolio_code_list)
        stock_end = portfolio[['Betavals', 'Vals', 'wght']].copy()

        index_bgn = db.future_data[(db.future_data['Trddt']==BGNDATE) & (db.future_data['Agmtcd']==AGREEMENT)]['Stprc'].values[0]
        index_end = db.future_data[(db.future_data['Trddt']==ENDDATE) & (db.future_data['Agmtcd']==AGREEMENT)]['Stprc'].values[0]

        self.stock_bgn = stock_bgn
        self.stock_end = stock_end
        self.index_bgn = index_bgn
        self.index_end = index_end


    def hedge(self):
        value_bgn = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        value_end = self.stock_end[self.stock_end['Betavals'] > self.flt]['Vals'].sum()

        spot_market = round(value_end - value_bgn, 3)
        index_market = round((self.index_bgn - self.index_end) * self.n * MULTIPLIER / 10000, 3)
        tran_cost = self.index_end * self.n * TRAN_RATE * MULTIPLIER / 10000
        result = round(spot_market + index_market - tran_cost, 3)

        print '\n| contracts\t| best h ratio\t| gain at spot\t| gain at index\t| result\t|'
        print '| {}\t| {}\t\t| {}\t| {}\t| {}\t|'.format(self.n,self.h, spot_market, index_market, result)


    def hedge_noprint(self):
        value_bgn = self.stock_bgn[self.stock_bgn['Betavals'] > self.flt]['Vals'].sum()
        value_end = self.stock_end[self.stock_end['Betavals'] > self.flt]['Vals'].sum()

        spot_market = round(value_end - value_bgn, 3)
        index_market = round((self.index_bgn - self.index_end) * self.n * MULTIPLIER / 10000, 3)
        tran_cost = self.index_end * self.n * TRAN_RATE * MULTIPLIER / 10000
        result = round(spot_market + index_market - tran_cost, 3)

        profit_n = round(spot_market / value_bgn, 3)
        profit_h = round(result / value_bgn, 3)

        return profit_n, profit_h




class VARModel(object):
    """ VAR Model"""
    def __init__(self, h=1.):
        self.h = h



class EMCModel(object):
    """ traditional Model"""
    def __init__(self, h=1.):
        self.h = h



class GARCHModel(object):
    """ traditional Model"""
    def __init__(self, h=1.):
        self.h = h



def portfolio_generate():
    portfolio_list = list()
    while len(portfolio_list) < 50:
        portfolio = sorted(random.sample(STOCK_LIST, 16))
        try:
            db = Database(portfolio)
            db.portfolio_init()
            if (len(db.portfolio)>=13 and (portfolio not in portfolio_list) and db.init_beta>0.8):
                portfolio_list.append(portfolio)
        except:
            pass

    return portfolio_list




if __name__ == '__main__':
    print 'Portfolio generating!'
    portfolio_list = portfolio_generate()
    print 'Portfolio generate finish!'

    db = Database()
    db.portfolio_init()

    portfolio, prtfl_bt, prtfl_value = db.get_daily_info(BGNDATE, db.portfolio_code_list)
    print '\n', '- ' * 30
    print portfolio
    print '{}, beta of portfolio is ({})'.format(BGNDATE, round(prtfl_bt, 4))
    portfolio, prtfl_bt, prtfl_value = db.get_daily_info(ENDDATE, db.portfolio_code_list)
    print '\n', '- ' * 30
    print portfolio
    print '{}, beta of portfolio is ({})'.format(ENDDATE, round(prtfl_bt, 4))

    print '\n', '- ' * 30
    print 'Traditional Model:'
    m1 = TraditionalModel()
    m1.hedge_init(db)

    # hedge for all contract / at beta > 1 / at beta > 0.8
    for i in [float("-inf"), 1.0, 0.8]:
        m1.flt = i
        m1.n_init()
        m1.hedge()

    # calc the He  He = sigma
    print '\ntesting with 50 portfolio!'
    for i in [float("-inf"), 1.0, 0.8]:
        sigma_n = list()
        sigma_h = list()
        for j in portfolio_list:
            db = Database(j)
            db.portfolio_init()

            m = TraditionalModel()
            m.hedge_init(db)
            m.flt = i
            m.n_init()
            profit_n, profit_h = m.hedge_noprint()
            sigma_n.append(profit_n)
            sigma_h.append(profit_h)
        sigma_n = np.var(sigma_n)
        sigma_h = np.var(sigma_h)

        He = round((sigma_n - sigma_h) / sigma_n, 3)
        print'He for (beta>{})\tis ({})'.format(i, He)


    print '\n', '- ' * 30
    print 'CAPM Beta Model:'
    m2 = CAPMBetaModel()
    m2.hedge_init(db)

    # hedge for all contract / at beta > 1 / at beta > 0.8
    for i in [float("-inf"), 1.0, 0.8]:
        m2.flt = i
        m2.n_init()
        m2.hedge()

    # calc the He  He = sigma
    print '\ntesting with 50 portfolio!'
    for i in [float("-inf"), 1.0, 0.8]:
        sigma_n = list()
        sigma_h = list()
        for j in portfolio_list:
            db = Database(j)
            db.portfolio_init()

            m = CAPMBetaModel()
            m.hedge_init(db)
            m.flt = i
            m.n_init()
            profit_n, profit_h = m.hedge_noprint()
            sigma_n.append(profit_n)
            sigma_h.append(profit_h)
        sigma_n = np.var(sigma_n)
        sigma_h = np.var(sigma_h)

        He = round((sigma_n - sigma_h) / sigma_n, 3)
        print'He for (beta>{})\tis ({})'.format(i, He)


    print '\n', '- ' * 30
    print 'VAR Model:'
    m3 = CAPMBetaModel()
    m3.hedge_init(db)

    # hedge for all contract / at beta > 1 / at beta > 0.8
    for i in [float("-inf"), 1.0, 0.8]:
        m3.flt = i
        m3.n_init()
        m3.hedge()

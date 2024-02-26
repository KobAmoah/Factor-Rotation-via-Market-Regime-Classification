"""
Created on Sat Feb  24 2024

    This program derives off initial code by Matthew Wang that sets up a class for Factor Investing with Market Regime 
Classification to be tested on Quantconnect. It differs from Matthew's implementation with the following changes:
1. I substitute the FamaFrench framework with an explicit focus on Value and Growth Factor investing. 
2. I apply sector neutralization to reduce the distributional effects of each factor in biasing the stock selection.
3. I incorporate quality in the Value Factor model to make it more robust to shifting economic cycles.
4. This application is long-only and is rebalanced monthly.
5. The created portfolio is score weighted to ensure it mimics appropriately its style propensity.

    The core hypothesis remains the same - to prove that a hidden markov model that rotates between factor models depending on
market conditions can perform better than the individual factor model themselves. 

original author: Matthew Wang
https://medium.com/@matthewwang_91639/algorithmic-factor-investing-with-market-regime-classification-6bc2f8c7168b

@author: Kobena Amoah
"""

import operator
from math import ceil,floor
import pandas as pd
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hmmlearn import hmm
from AlgorithmImports import *

class HMMHybrid(QCAlgorithm):

    def Initialize(self):
        #Switch value for each regime
        self.switch = 'neutral'
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        spy = self.AddEquity("SPY", Resolution.Daily)
        self.reference = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.SetStartDate(2013, 12, 31)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)         
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose("SPY"), self.MarketClose)
        self.Schedule.On(self.DateRules.MonthStart("SPY"), \
                 self.TimeRules.AfterMarketOpen("SPY"), \
                 self.Reset)
        self.daily_return = 0
        self.prev_value = self.Portfolio.TotalPortfolioValue
        self.numberOfSymbols = 500
        self.symbols = [ spy.Symbol ]
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.AfterMarketOpen("SPY"), Action(self.rebalance))
    
    def Reset(self):
        if self.switch == 'bear':
            self.Value()
        else:
            self.GrowthModel()

    def CoarseSelectionFunction(self, coarse):
        CoarseWithFundamental = [x for x in coarse if x.HasFundamentalData and (float(x.Price)>1)]
        
        sortedByDollarVolume = sorted(CoarseWithFundamental, key=lambda x: x.DollarVolume, reverse=True)
        top = sortedByDollarVolume[:self.numberOfSymbols]
        return [i.Symbol for i in top]

    def FineSelectionFunction(self, fine):
        # FINE FILTERING FOR VALUE STOCKS

        # drop stocks which don't have the information we need.
        # you can try replacing those factors with your own factors here

        filtered_fine = [x for x in fine if (x.ValuationRatios.EVtoEBIT) > 0
                                        and x.ValuationRatios.PBRatio
                                        and x.MarketCap 
                                        and (x.ValuationRatios.FCFYield) > 0
                                        and (x.ValuationRatios.EarningYield) > 0
                                        and x.OperationRatios.LongTermDebtEquityRatio.Value
                                        and x.OperationRatios.OperationMargin.Value]

        
        # Create a DataFrame with Stock List and MorningstarSectorCode columns
        Value_data = {'Stock': [x.Symbol for x in filtered_fine],
                    'MorningstarSectorCode': [x.AssetClassification.MorningstarSectorCode for x in filtered_fine],
                    'FCFYield': [x.ValuationRatios.FCFYield for x in filtered_fine],
                    'PBRatio': [x.ValuationRatios.PBRatio for x in filtered_fine],
                    'MarketCap': [x.MarketCap for x in filtered_fine],
                    'EarningYield':[x.ValuationRatios.EarningYield for x in filtered_fine],
                    'EVtoEBIT':[x.ValuationRatios.EVtoEBIT for x in filtered_fine],
                    'LongTermDebtEquityRatio':[x.OperationRatios.LongTermDebtEquityRatio.Value for x in filtered_fine],
                    'OperationMargin':[x.OperationRatios.OperationMargin.Value for x in filtered_fine]}

        df_value = pd.DataFrame(Value_data)
    
        # Group by sector for Value stocks
        value_grouped = df_value.groupby('MorningstarSectorCode')

        value_factors = ['PBRatio', 'MarketCap','EarningYield','EVtoEBIT','FCFYield','LongTermDebtEquityRatio','OperationMargin']
        self.value_long = []
        self.value_score = np.array()
        for sector, group in value_grouped:
            factor_scores = pd.DataFrame()

            for factor in value_factors:
                if factor not in ['MarketCap']:
                    group[factor] = (group[factor]- group[factor].mean())/group[factor].std()
                    rank = group[factor].rank(ascending=True)
                    factor_scores[factor + '_Score'] = rank
                else: 
                    group[factor] = (group[factor]- group[factor].mean())/group[factor].std()
                    rank = group[factor].apply(np.log).rank(ascending=True)
                    factor_scores[factor + '_Score'] = rank
                    

            # Combine scores for each factor
            group['Total_Score'] = factor_scores.sum(axis=1)
            group_sorted = group.sort_values(by='Total_Score')

            # Select the highest decile stocks
            decile_threshold = int(len(group_sorted)*0.1)
            selected_group = group_sorted.iloc[:decile_threshold]
            self.value_long.extend(selected_group['Stock'])
            self.value_score.append(selected_group['Total_Score'])
            
        # FINE FILTERING FOR GROWTH STOCKS
        filtered_fine_growth = [x for x in fine if x.EarningReports.TotalDividendPerShare.ThreeMonths
                                                and ((x.OperationRatios.ROE.Value) *100) > 0
                                                and x.EarningReports.BasicEPS.TwelveMonths
                                                and (x.OperationRatios.RevenueGrowth.Value) * 100]

        # Create a DataFrame with Stock List and MorningstarSectorCode columns for Growth stocks
        growth_data = {'Stock': [x.Symbol for x in filtered_fine_growth],
                    'MorningstarSectorCode': [x.AssetClassification.MorningstarSectorCode for x in filtered_fine_growth],
                    'TotalDividendPerShare.ThreeMonths': [x.EarningReports.TotalDividendPerShare.ThreeMonths if hasattr(x.EarningReports, 'TotalDividendPerShare') else 0 for x in filtered_fine_growth],
                    'ROE.Value': [x.OperationRatios.ROE.Value for x in filtered_fine_growth],
                    'RevenueGrowth': [x.OperationRatios.RevenueGrowth.Value for x in filtered_fine_growth],
                    'BasicEPS.TwelveMonths': [x.EarningReports.BasicEPS.TwelveMonths for x in filtered_fine_growth]}
        df_growth = pd.DataFrame(growth_data)


        # Group by sector for Growth stocks
        growth_grouped = df_growth.groupby('MorningstarSectorCode')
        factors_growth = ['TotalDividendPerShare.ThreeMonths', 'ROE', 'RevenueGrowth','BasicEPS.TwelveMonths']

        self.growth_long = []
        self.growth_score = np.array()

        for sector, group_growth in growth_grouped:
            factor_scores_growth = pd.DataFrame()

            for factor_growth in factors_growth:
                if factor_growth not in [ 'ROE','RevenueGrowth']:
                    group_growth[factor_growth] = (group_growth[factor_growth] - group_growth[factor_growth].mean())/group_growth[factor_growth].std()
                    group_growth[factor_growth] = group_growth[factor_growth].rank(ascending=True)
                    factor_scores_growth[factor_growth + '_Score'] = group_growth[factor_growth]
                else:
                    group_growth[factor_growth] = group_growth[factor_growth].rank(ascending=True)
                    factor_scores_growth[factor_growth + '_Score'] = group_growth[factor_growth]



            # Combine scores for each factor
            group_growth['Total_Score'] = factor_scores_growth.sum(axis=1)
            group_sorted_growth = group_growth.sort_values(by='Total_Score', ascending=False)

            # Select the highest decile stocks
            decile_threshold_growth = int(len(group_sorted_growth)*0.1)
            selected_group_growth = group_sorted_growth.iloc[:decile_threshold_growth]
            self.growth_long.extend(selected_group_growth['Stock'])
            self.growth_score.append(selected_group['Total_Score'])

        if self.switch == 'bear':
            return self.value_long , self.value_score
        else:
            return self.growth_long , self.growth_score


    def OnData(self, data):
        pass

    def rebalance(self):
        next = self.next = self.train()
        if self.Portfolio.TotalHoldingsValue == 0:
            self.switch = next
            if self.switch == 'bear':
                self.Value()
            else:
                self.GrowthModel()
            return
        
        if next == self.switch:
            return
            
        self.switch = next
        
        if next == 'neutral':
            return
            
        # Assign each stock equally.
        if self.switch == 'bear':
            self.Value()
        else:
            self.GrowthModel()

    def Value(self):
        #self.Log("Value")
        for kvp in self.Portfolio:
            if kvp.Value.Invested and not (kvp.Key in self.value_long):
                self.SetHoldings(kvp.Key, 0)
        for symbol, score in zip(self.value_long, self.value_score):
            self.SetHoldings(symbol, score / np.sum(self.value_score))

    def GrowthModel(self):
        #self.Log("Growth Model")
        for kvp in self.Portfolio:
            if kvp.Value.Invested and not kvp.Key in self.growth_long:
                self.SetHoldings(kvp.Key, 0)
        for symbol, score in zip(self.growth_long, self.growth_score):
            self.SetHoldings(symbol, score / np.sum(self.growth_score))

    def MarketClose(self):
        self.daily_return = 100*((self.Portfolio.TotalPortfolioValue - self.prev_value)/self.prev_value)
        self.prev_value = self.Portfolio.TotalPortfolioValue
        self.Log(self.daily_return)
        #self.Log("Switch: {}".format(self.switch))
        return

    def train(self):
        # Hidden Markov Model Modifiable Parameters
        hidden_states = 3;
        em_iterations = 75;
        data_length = 2520;
        # num_models = 7;

        #history = self.History("SPY", 2718, Resolution.Daily)
        #prices = list(history.loc["SPY"]['close'])

        history = self.History(self.symbols, data_length, Resolution.Daily)
        for symbol in self.symbols:
            if not history.empty:
                # get historical open price
                prices = list(history.loc[symbol.Value]['close'])

        # Volatility is computed by obtaining variance between current close and
        # prices of past 30 days
        Volatility = []

        # MA is the 21 day SMA
        MA = []

        # Return is the single-day percentage return
        Return = []
        ma_sum = 0;

        # Warming up data for moving average and volatility calculations
        for i in range (0, 21):
            Volatility.append(0);
            MA.append(0);
            Return.append(0);
            ma_sum += prices[i];
        # Filling in data for return, moving average, and volatility
        for i in range(0, len(prices)):
            if i >= 21:
                tail_close = prices[i-21];
                prev_close = prices[i-1];
                head_close = prices[i];
                ma_sum = (ma_sum - tail_close + head_close);
                ma_curr = ma_sum/21;
                MA.append(ma_curr);
                Return.append(((head_close-prev_close)/prev_close)*100);
                #Computing Volatility
                vol_sum = 0;
                for j in range (0, 21):
                    curr_vol = abs(ma_curr - prices[i-j]);
                    vol_sum += (curr_vol ** 2);
                Volatility.append(vol_sum/21);

        prices = prices[21:]
        Volatility = Volatility[21:]
        Return = Return[21:]

        # Creating the Hidden Markov Model
        model = hmm.GaussianHMM(n_components = hidden_states,
                                covariance_type="full", n_iter = em_iterations);

        obs = [];
        for i in range(0, len(Volatility)):
            arr = [];
            arr.append(Volatility[i]);
            arr.append(Return[i]);
            obs.append(arr);

        # Fitting the model and obtaining predictions
        model.fit(obs)
        predictions = model.predict(obs)

        # Regime Classification
        regime_vol = {};
        regime_ret = {};

        for i in range(0, hidden_states):
            regime_vol[i] = [];
            regime_ret[i] = [];

        for i in range(0, len(predictions)):
            regime_vol[predictions[i]].append(Volatility[i]);
            regime_ret[predictions[i]].append(Return[i]);

        vols = []
        rets = []
        today_regime = predictions[-1]
        for i in range(0, hidden_states):
            vol_dist = Distribution()
            vol_dist.Fit(regime_vol[i])
            vols.append(vol_dist.PDF(Volatility[-1]))
            ret_dist = Distribution()
            ret_dist.Fit(regime_ret[i])
            rets.append(ret_dist.PDF(Return[-1]))

        # > 0.3 Low-Pass Filter
        bear = -1
        bull = -1
        thresh_return = 0.5
        low_vol = 100
        for i in range(0,hidden_states):
            if sum(regime_ret[i])/len(regime_ret[i])< -thresh_return:
                bear = i
            if sum(regime_ret[i])/len(regime_ret[i]) > thresh_return:
                bull = i
            
        if vols[today_regime] / sum(vols) >= 0.3 and abs(rets[today_regime] / sum(rets)) >= 0.3:
            if bear == today_regime:
                return 'bear'
            else:
                return 'bull'
        else:
            return 'neutral'

# Kolmogorov-Smirnov Test to find best distribution
class Distribution(object):

    def __init__(self, dist_names_list = []):
        self.dist_names = ['norm','lognorm','expon', 'gamma',
                           'beta', 'rayleigh', 'norm', 'pareto']
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None
        self.isFitted = False


    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)
            self.params[dist_name] = param
            #Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param);
            self.dist_results.append((dist_name,p))

        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results,key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p
        self.isFitted = True

        return self.DistributionName, self.PValue

    def PDF(self, x):
        dist = getattr(scipy.stats, self.DistributionName)
        n = dist.pdf(x, *self.params[self.DistributionName])
        return n

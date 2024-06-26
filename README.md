![GitHub all releases](https://img.shields.io/github/downloads/KobAmoah/Factor-Rotation-via-Market-Regime-Classification/total)
![GitHub language count](https://img.shields.io/github/languages/count/KobAmoah/Factor-Rotation-via-Market-Regime-Classification) 
![GitHub top language](https://img.shields.io/github/languages/top/KobAmoah/Factor-Rotation-via-Market-Regime-Classification?color=yellow) 
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/KobAmoah/Factor-Rotation-via-Market-Regime-Classification)
![GitHub forks](https://img.shields.io/github/forks/KobAmoah/Factor-Rotation-via-Market-Regime-Classification?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/KobAmoah/Factor-Rotation-via-Market-Regime-Classification?style=social)

# Factor-Rotation-via-Market-Regime-Classification
The following adapts an implementation by ["Matthew Wang"](https://medium.com/@matthewwang_91639/algorithmic-factor-investing-with-market-regime-classification-6bc2f8c7168b) who proposes an investment strategy that leverages statistical modeling to take advantage of volatility clustering in financial time series data. He focuses on identifying different market regimes, i.e a bear, bull or neutral market, using a Hidden Markov Model (HMM) and then adapting the investing model to the current market regime - although his implementation was confined to switching between Momentum-type Growth model and the Fama-French 3-factor model.

However, the Fama-French framework's plight is well documented which Robert D. Arnott and his co-authors(2021) attribute to poor factor definitions. To cater for poor factor definitions among other issues, I modify Matthew's his orignal framework to incorporate the following:

1. I  make more explicit Value and Growth Factors
2. I apply sector neutralization to reduce the distributional effects of each factor in biasing the stock selection.
3. I incorporate Quality into the Value Factor model as the implicit goal of the rotation is to deliver a factor framework that does well over all economic cycles.
4. I substitute Matthew's original idea which is Leverage dependent, applied in a long-short framework and is rebalanced daily, to an application that is long -only and is is rebalanced only when the model detects a shift in economic conditions.

The idea can be applied not only at the institutional level, but at the retail(individual) level as well. The framework is designed to take into account accumulating tax costs on performance.

### Theory:
- Volatility Clustering: Financial time series data often exhibits volatility clustering, where the variance of returns persists over time.
- Motivation: The 2020 market turmoil due to the Coronavirus inspired the idea of designing a strategy that adapts to different market regimes.
- Hypothesis: A hidden markov investing model rotating between factor models based on market conditions can outperform individual factor models.

### Gaussian HMM:
- Parameters: Transition matrix (A), observation probability distributions (µ and σ), and initial starting states vector (π).
- Observation Variables: Volatility on the 21-day moving average and daily S&P500 returns.
- Objective: Train HMM to identify market regimes and adapt an overarching trading model based on the current regime.

### Applying HMM:
- Subproblems: Estimate occurrence probability, find optimal sequence of hidden states, and determine optimal parameters.
- Algorithms Used: Forward algorithm, Viterbi algorithm, and Baum-Welch algorithm.
- Implementation: Used Python hmmlearn library to create and train a Gaussian HMM on data from December 2003 to December 2013.

### Methodology:
- Factor Models: Tested Value and Growth factor models.
- Performance Evaluation: Used QuantConnect’s backtesting platform over a 10-year period. (from December 2013 to December 2023)
- Observation: Growth model performed well in bull markets, but suffered in downturns; Value factor models performed better in downturns.

### Strategy Basics:
- Regime Detection: Used a sliding window of 2520 points to train the HMM daily for regime detection.
- Regime Switching: Switched between Growth and Value factor models based on detected regimes.
- Regime Confirmation: Implemented a high-pass filter using the Kolmogorov-Smirnov test to confirm regime shifts before switching models.

### Performance:
The following is the strategy's performance over a 10 year window from 2013-12-31 to 2023-12-31. The SPY returned an annualized performance of 12.7% over the period with the strategy returning 18.26% on an annualized basis.

<img width="736" alt="Factor Rotation 1" src="https://github.com/KobAmoah/Factor-Rotation-via-Market-Regime-Classification/assets/108365002/a6003965-0232-4b38-a664-dcc6f5a8f7ea">

<img width="723" alt="Factor Rotation 2" src="https://github.com/KobAmoah/Factor-Rotation-via-Market-Regime-Classification/assets/108365002/c6620caa-1e43-442c-9798-da3857a33228">

<img width="722" alt="Factor Rotation 3" src="https://github.com/KobAmoah/Factor-Rotation-via-Market-Regime-Classification/assets/108365002/861f890b-8a8a-44ea-af75-72f2ed19109d">


### Conclusion:
The strategy aims to optimize returns by adapting to different market regimes using a Hidden Markov Model, with a focus on switching between a Growth model and a Value factor model based on detected market conditions. The high-pass filter helps confirm regime shifts before making portfolio adjustments.

## Credits
I am incredibly grateful to Matthew Wang for his research on the topic. His idea of dynamically adapting a portfolio between factor models using a Hidden Markov Model, which classifies market regimes (bear, bull, or neutral) inspired this project. The referenced paper by Robert D. Anott et al. can be accessed at ["Reports of Value’s Death May Be Greatly Exaggerated"](https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1842704)

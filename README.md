![GitHub all releases](https://img.shields.io/github/downloads/KobAmoah/Factor-Rotation-via-Market-Regime-Classification/total)
![GitHub language count](https://img.shields.io/github/languages/count/KobAmoah/Factor-Rotation-via-Market-Regime-Classification) 
![GitHub top language](https://img.shields.io/github/languages/top/KobAmoah/Factor-Rotation-via-Market-Regime-Classification?color=yellow) 
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/KobAmoah/Factor-Rotation-via-Market-Regime-Classification)
![GitHub forks](https://img.shields.io/github/forks/KobAmoah/Factor-Rotation-via-Market-Regime-Classification?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/KobAmoah/Factor-Rotation-via-Market-Regime-Classification?style=social)

# Factor-Rotation-via-Market-Regime-Classification
The following adapts an implementation by ["Matthew Wang"](https://medium.com/@matthewwang_91639/algorithmic-factor-investing-with-market-regime-classification-6bc2f8c7168b) who proposes an investment strategy that leverages statistical modeling to take advantage of volatility clustering in financial time series data. He focuses on identifying different market regimes using a Hidden Markov Model (HMM) and then adapting the investing model based on the current market regime - although his implementation was confined to switching between Momentum-type Growth model and the Fama-French 3-factor model.

However, the Fama-French framework's plight is well documented which Robert D. Arnott and his co-authors attribute to poor factor definitions. To cater for this, I adapt the original code and make more explicit the Value and Growth Factors as well as apply sector neutralization to reduce the distributional effects of each factor in biasing the stock selection. I also incorporate Quality into the Value Factor model as the implicit goal of rotation is to deliver a factor framework that does well over all economic cycles. Lastly, Matthew Wang's original idea is dependent on Leverage applied in a long-short framework which is rebalanced daily. This application is long only and is rebalanced monthly.

The idea is targeted as an institutional strategy, however, for retail investors(individuals), it should be noted monthly trading will lead to accumulating tax costs. To reduce the tax impact of trading, one can modify rebalance rule to only rebalance when the signal changes or after the signal fails to change for a predefined number of months.

*** I have not fully backtested the code in Quantconnect yet, but do hope to do so at some point. 

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
- Performance Evaluation: Used QuantConnect’s backtesting platform over a 10-year period.
- Observation: Growth model performed well in bull markets, but suffered in downturns; Value factor models performed better in downturns.

### Strategy Basics:
- Regime Detection: Used a sliding window of 2520 points to train the HMM daily for regime detection.
- Regime Switching: Switched between Growth and Value factor models based on detected regimes.
- Regime Confirmation: Implemented a high-pass filter using the Kolmogorov-Smirnov test to confirm regime shifts before switching models.

### Conclusion:
The strategy aims to optimize returns by adapting to different market regimes using a Hidden Markov Model, with a focus on switching between a Growth model and a Value factor model based on detected market conditions. The high-pass filter helps confirm regime shifts before making portfolio adjustments.

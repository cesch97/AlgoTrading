AlgoTrading
-----------
This project aims to combine Neural Networks and Genetic programming to create a trading algortihm able to perform consistently in different market conditions.  
To achive this result we divide the problem in two steps:
1. First we train a set of **strategies** to trade on a specific market.
2. Then we train a **trading system** which select different strategies and dynamically allocate capital to each strategy based on broad market conditions.

### Strategies
A strategy is composed of two layers:
1. **Feature extraction**: This layer uses a customized version of *Cartesian Genetic Programming* to extract features from the market data. It does so by combining techinal indicators and selectors, technical indicators are just the usual indicators used in technical analysis, while selectors are used to select and perform operations on a subset of the indicators to create new features.
2. **Trading logic**: This layer uses a Neural Network to decide when to buy and sell. The input of the Neural Network is the features extracted by the previous layer and the output is the decision to buy, sell or hold. The neural newtrok is tradined with reinforcement learning using an evolution strategy.

Both layers are combined and trained concurrently by simulating the trading on historical data, the objective is a custom metric that combines the return and the drawdown of the strategy, to incentivize the strategy to produce consistent returns.

### Trading System
A trading system is composed of two layers:
1. **Feature extraction**: Similar to the strategy, this layer uses a customized version of *Cartesian Genetic Programming* to extract features from the market data. The difference is that the features are extracted from different markets.
2. **Capital allocation**: This layer uses a Neural Network to decide how to allocate capital to each strategy. The input of the Neural Network is the features extracted by the previous layer and the output is the allocation of capital to each strategy. The neural newtrok is tradined with reinforcement learning using an evolution strategy.

The training is similar to the strategy, the objective is to maximize the return of the trading system while keeping, drawdown and volatility low.

### Training
The training happens in two stages:
1. **Strategy training**: A set of stretgies is trained on a specific market, the training is done by simulating the trading on historical data. The top performing strategies are selected.
2. **Trading system training**: A trading system is trained using the top performing strategies from the previous step.

The detailed explanation of how to train a neural network using evolution startegies can be found [here](https://github.com/cesch97/NeuroEvolution)

The project is optimized to run on a large number of cores, with a main process resposible for the overall training and a set of worker processes responsible for the simulations thus calculating the fitness of the strategies and the trading system. I tested the project succescfully on a 256 core machine.

### Deployment
After the training is complete the trading system is ready to be deployed for live trading and backtesting. The project contains a simple server that can recive market data and send back the trading signals. I was able to integrate project with the "CTrader" a trading platform that allows to create custom trading bots in C#.

### Results
The project was able to consistently produce trading systems capable of producing high and consistent profits in backtesting, however the results in live trading were not as good. Despite having carefully design the fitness metric to prevent overfitting the limited amount of data was likely the cause of the poor results. 

### Usage
To train a pool of strategies:
1. Create a configuration file with the parameters of the training (see `configs/strat_creation.yml` for an example)
2. Run `julia train_strat.jl configs/strat_creation.yml`

To train a trading system:
1. Create a configuration file with the parameters of the training (see `configs/trad_sys_creation.yml` for an example)
2. Run `julia train_trad_sys.jl configs/trad_sys_creation.yml`

To evaluate a trading system:
1. Create a configuration file with the parameters of the evaluation (see `configs/trad_sys_evaluation.yml` for an example)
2. Run `julia utils.jl evaluate configs/trad_sys_evaluation.yml`

To serve a trading system:
1. Create a configuration file with the parameters of the deployment (see `configs/trad_sys_serving.yml` for an example)
2. Run `julia utils.jl serve configs/trad_sys_serving.yml`

To clean the pool of strategies and take only the top performing ones:
1. Create a configuration file with the parameters of the cleaning (see `configs/strat_clean.yml` for an example)
2. Run `julia utils.jl clean-strat configs/strat_clean.yml`

To clean the pool of trading systems and take only the top performing ones:
1. Create a configuration file with the parameters of the cleaning (see `configs/trad_sys_clean.yml` for an example)
2. Run `julia utils.jl clean-trad-sys configs/trad_sys_clean.yml`

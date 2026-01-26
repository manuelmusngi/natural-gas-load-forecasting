🔥 Natural Gas Load Forecasting with Enhanced LSTM & Hybrid Deep Learning Models
A Research‑Backed, Production‑Ready End‑to‑End Project

📘 1. Project Summary

This project implements a short‑term natural gas load forecasting system using state‑of‑the‑art deep learning and hybrid ensemble architectures. It integrates:

- Enhanced LSTM with feature optimization + incremental learning  (Wang et al., 2025)
- Hybrid Ensemble + VMD + Transformer–ResLSTM  (Zhao et al., 2024)
- Feature‑rich gas load forecasting frameworks  (2023–2024 LSTM studies)

The system forecasts hourly or daily natural gas consumption for utilities, LDCs, and gas‑fired power generators using:

- Weather features
- Calendar effects
- Lagged gas load
- Multi‑scale decomposed signals
- Incremental model updates

The architecture is designed for operational deployment, research benchmarking, and scalable experimentation.

🧠 2. Research Foundations

Wang et al. (2025)
Short‑Term Natural Gas Consumption Forecasting Using Enhanced LSTM with Feature Optimization and Incremental Learning

- Introduces feature‑optimized LSTM
- Uses incremental learning to adapt to new data
- Shows improved robustness during regime shifts (weather shocks, holidays)

Zhao et al. (2024)
Hybrid Ensemble–Deep Learning Model (EL + VMD + Transformer–ResLSTM)

- Combines:
    - Empirical Mode Decomposition (VMD)
    - Transformer encoder
    - Residual LSTM
- Captures multi‑scale temporal patterns
- Outperforms ARIMA, GBM, vanilla LSTM
- LSTM Gas Forecasting Studies (2023–2024)
  
- Stress importance of:

  - Weather integration
  - Feature engineering
  - Multi‑step sequence modeling
  - Hybrid decomposition + deep learning
  
🏗️ 3. Project Architecture (Production‑Ready)

gas-load-forecasting/\
│
├── config/\
│   ├── [data_config.yaml](https://github.com/manuelmusngi/ng-load-forecasting/blob/main/config/data_config.yaml)\
│   ├── [model_config.yaml](https://github.com/manuelmusngi/ng-load-forecasting/blob/main/config/model_config.yaml)\
│   └── [training_config.yaml](https://github.com/manuelmusngi/ng-load-forecasting/blob/main/config/training_config.yaml)\
│
├── data/\
│
├── notebooks/\
│   ├── EDA.ipynb\
│   ├── Feature_Engineering.ipynb\
│   └── Model_Benchmarking.ipynb\
│
├── src/\
│   ├── data/\
│   │   ├── [load_data.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/data/load_data.py)\
│   │   ├── [preprocess.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/data/preprocess.py)\
│   │   └── [feature_engineering.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/data/feature_engineering.py)\
│   │
│   ├── models/\
│   │   ├── [lstm_enhanced.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/models/lstm_enhanced.py)\
│   │   ├── [transformer_reslstm.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/models/transformer_reslstm.py)\
│   │   ├── [vmd_decomposition.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/models/vmd_decomposition.py)\
│   │   ├── [ensemble.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/models/ensemble.py)\
│   │   └── [baselines.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/models/baselines.py)\
│   │
│   ├── training/\
│   │   ├── [train_lstm.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/training/train_lstm.py)\
│   │   ├── [train_hybrid.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/training/train_hybrid.py)\
│   │   └── [incremental_update.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/training/incremental_update.py)\
│   │
│   ├── evaluation/\
│   │   ├── [metrics.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/evaluation/metrics.py)\
│   │   └── [backtest.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/evaluation/backtest.py)\
│   │
│   ├── utils/\
│   │   ├── logger.py\
│   │   ├── config_parser.py\
│   │   └── plotting.py\
│   │
│   └── pipeline/\
│       ├── [forecasting_pipeline.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/pipeline/forecasting_pipeline.py)\
│       └── [incremental_pipeline.py](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/src/pipeline/incremental_pipeline.py)\
│
├── [main.py](https://github.com/manuelmusngi/ng-load-forecasting/blob/main/main.py)\
├── [requirements.txt](https://github.com/manuelmusngi/natural-gas-load-forecasting/blob/main/requirements.txt)\
└── README.md



#### License
This project is licensed under the [MIT License](https://github.com/manuelmusngi/regime_switching_models/edit/main/LICENSE).


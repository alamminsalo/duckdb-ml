# Machine Learning DuckDB Extension

This repository contains a Rust extension for the DuckDB database, adding machine learning capabilities through custom scalar functions. The extension provides functionalities to create, list, train, and predict models using neural networks.

## Table of Contents
| Function | Inputs | Description |
|----------|--------|-------------|
| `ml_create` | modelname: VARCHAR, spec: VARCHAR | Creates a new machine learning model with the given name and specification. |
| `ml_list` |  | Lists all registered models. |
| `ml_train` | modelname: VARCHAR, features: LIST(FLOAT), targets: LIST(FLOAT) -> LIST(FLOAT) | Trains a machine learning model using the specified features and targets for the given model name. Returns predicted outputs after training the model. |
| `ml_pred` | modelname: VARCHAR, features: LIST(FLOAT) -> LIST(FLOAT) | Predicts outcomes using the trained model with the specified name and input features. |

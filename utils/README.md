# Utils

This folder contains useful code to execute the models and make an experimental framework.

- informer_dataset.py: this is an implementation of the informer dataset adapted to TSFEDL and our implementation of transformer.
- metrics_informer.py: adaptation of the informer metrics to our framework.
- informer_timefeature.py: time utils to manage timestamps necessary to use informer_dataset.
- informer_tools.py: informer tools, currently only an online scaler. This could rely on sklearn.
- progress_bar.py: pytorch callback to implement a custom progress bar.
- timeseries_dataset.py: time series dataset to manage TiNA dataset.
- tsfedl_top_module.py: implementation of a top module for each of the models from TSFEDL. These models require a last module used after the CNN+LSTM architecture and must be configured by the user.

https://github.com/zhouhaoyi/Informer2020
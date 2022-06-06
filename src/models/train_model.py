"""
Author: Ilia Altmark
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, metrics, Pool
import optuna
from optuna.samplers import TPESampler
from catboost.utils import eval_metric

X_train = pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/X_train.csv')
y_train = np.ravel(pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/y_train.csv',
    index_col=0))

X_val = pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/X_val.csv')
y_val = np.ravel(pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/y_val.csv',
    index_col=0))

X_test = pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/X_test.csv')
y_test = np.ravel(pd.read_csv(
    '/content/drive/MyDrive/Colab Notebooks/Spotify DA/Data/y_test.csv',
    index_col=0))

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
test_pool = Pool(X_test, y_test)


def calc_test_quality(**kwargs):
    model = CatBoostClassifier(**kwargs)
    model.fit(train_pool, verbose=0, eval_set=val_pool)
    y_pred = model.predict_proba(test_pool)
    return model, eval_metric(test_pool.get_label(), y_pred[:, 1], 'AUC')


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'boosting_type': trial.suggest_categorical('boosting_type',
                                                   ['Ordered', 'Plain']),
        'iterations': 3000,
        'random_seed': 42,
        'eval_metric': metrics.AUC(),
        'logging_level': 'Silent',
        'use_best_model': True,
        'od_type': 'Iter',
        'od_wait': 200
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=0, eval_set=val_pool)
    y_pred = model.predict_proba(val_pool)
    return eval_metric(val_pool.get_label(), y_pred[:, 1], 'AUC')


def main():
    sampler = TPESampler(seed=123)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=70)

    best_params = {
        'depth': study.best_params['depth'],
        'boosting_type': study.best_params['boosting_type'],
        'l2_leaf_reg': study.best_params['l2_leaf_reg'],
        'learning_rate': study.best_params['learning_rate'],
        'iterations': 3000,
        'random_seed': 42,
        'eval_metric': metrics.AUC(),
        'logging_level': 'Silent',
        'use_best_model': True,
        'od_type': 'Iter',
        'od_wait': 200
    }

    # best params:
    # ([0.653434624198804],
    #  {'boosting_type': 'Plain',
    #   'depth': 6,
    #   'l2_leaf_reg': 7.368808704421745,
    #   'learning_rate': 0.012446929729117526})

    opt_catboost, auc = calc_test_quality(**best_params)
    print(auc, study.best_params)


if __name__ == "__main__":
    main()

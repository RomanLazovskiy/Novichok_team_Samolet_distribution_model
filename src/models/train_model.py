from sklearn.metrics import classification_report
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from category_encoders import CatBoostEncoder

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    log_loss,
    )

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from eli5.sklearn import PermutationImportance
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
from copy import deepcopy

def evaluate_metrics(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred) 
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int)) 
    precision = precision_score(y_true, (y_pred > 0.5).astype(int)) 
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int)) 
    # confusion = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    pr_auc = average_precision_score(y_true, y_pred) 
    logloss = log_loss(y_true, y_pred)

    print(f"ROC AUC: {roc_auc}", end='; ')
    print(f"Accuracy: {accuracy}", end='; ')
    print(f"Precision: {precision}", end='; ')
    print(f"Recall: {recall}", end='; ')
    print(f"F1 Score: {f1}", end='; ')
    print(f"PR AUC: {pr_auc}", end='; ')
    print(f"Log Loss: {logloss}")

    
class FeatureSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, features):
        """
        Класс для отбора заданных признаков.
        
        Параметры:
        features (list): Список заданных признаков.
        """
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Метод для отбора заданных признаков из датасета.
        
        Параметры:
        df (pd.DataFrame): Входной датасет.
        
        Возвращает:
        pd.DataFrame: Датасет, содержащий только отобранные признаки.
        
        Исключения:
        ValueError: Генерируется, если какие-то из заданных признаков отсутствуют во входном датасете.
        """
        # Проверяем наличие всех заданных признаков во входном датасете
        missing_features = set(self.features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Признаки {missing_features} отсутствуют во входном датасете.")
        
        # Отбираем заданные признаки из датасета
        selected_df = X[self.features]
        
        return selected_df

def objective(trial, trf, df_train, df_val, target_col='target'):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'min_sum_hessian_in_leaf ': trial.suggest_float('min_sum_hessian_in_leaf', 0.0, 10.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True]),
        'random_state': trial.suggest_categorical('random_state', [42])
    }
    pruning_callback = LightGBMPruningCallback(trial, "auc")
    trf = trf.fit(df_train, df_train[target_col])

    model = lgb.train(params, lgb.Dataset(trf.transform(df_train), label=df_train[target_col]), num_boost_round=1000,
                    valid_sets=[lgb.Dataset(trf.transform(df_val), label=df_val[target_col])], 
                    early_stopping_rounds=100, verbose_eval=False, callbacks=[pruning_callback])
   
    y_pred = model.predict(trf.transform(df_val))

    evaluate_metrics(df_val[target_col], y_pred)
    roc_auc = roc_auc_score(df_val[target_col], y_pred) 

    return roc_auc

def split_dataset_by_target(data, target, n):
    """
    Разделить датасет на n групп, с одинаковым положительным таргетом и разными негативными.
    :param data: pandas.DataFrame, содержащий данные для разделения.
    :param target: str, описывающий положительный таргет.
    :param n: количество групп для разделения (и, соответственно, количество моделей).
    :return: список, содержащий n кортежей из данных для обучения.
    """
    sorted_data = data.sort_values(target, kind='mergesort')
    positive_set_index = sorted_data.index[sorted_data[target] == 1]
    positive_set_size = len(positive_set_index)
    negative_set_index = sorted_data.index[sorted_data[target] == 0]
    negative_set_size = len(negative_set_index)
    negative_sets = []
    for i in range(n-1):
        negative_sets.append(negative_set_index[i*(negative_set_size//n):(i+1)*(negative_set_size//n)])
    negative_sets.append(negative_set_index[(n-1)*(negative_set_size//n):])
    splitted_data = []
    for i in range(n):
        negative_index = negative_sets[i]
        train_data_index = positive_set_index.append(negative_index)
        # test_data_index = negative_set_index.difference(negative_index)
        train_data = data.loc[train_data_index]
        # test_data = data.loc[test_data_index]
        splitted_data.append(train_data)
    return splitted_data
    
class ModelBuilder(object):
    def __init__(self, cat_cols, features_name, ml_cfg, target_col='target', params=None):
        
        self.features = features_name
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.cfg = ml_cfg
        selector = FeatureSelector(features=features_name)
        enc = CatBoostEncoder(cols=cat_cols)
        lgbm = LGBMClassifier()
        self.model = Pipeline([('sel', selector), ('enc', enc), ('lgbm', lgbm)])      
        self.model.set_params(**params) 
        
        
    def train(self, df_train):
        model = self.model.fit(df_train, df_train[self.target_col])
        return model
        
    def select_features(self, df_train, df_val):

        model = self.train(df_train)
        transformer = Pipeline(model.steps[:-1])
        model = model.steps[-1][-1]
        transformed_val_data = transformer.transform(df_val)

        print('start feature select...')
        perm = PermutationImportance(model, random_state=42) \
                                    .fit(transformed_val_data, df_val[self.target_col])
        print('end feature select...')

        feature_importances = list(zip(transformed_val_data.columns, perm.feature_importances_))
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        selected_featurs = [name_col for name_col, value in feature_importances if value > 0]
        print(f'selected {len(selected_featurs)} out of {len(df_train.columns)}')
        
        cat_cols = list(set(self.cat_cols) & set(selected_featurs))
        selected_features = {'sel__features': selected_featurs,'enc__cols': cat_cols}
        return selected_features
    
    def params_tuning(self, df_train, df_val, model):
        print('start tuning...')
        study = optuna.create_study(direction='maximize')
        transformer = Pipeline(model.steps[:-1])
        study.optimize(lambda trial: objective(trial, transformer, df_train, df_val),
                       n_jobs=1, n_trials=self.cfg['n_trails'])
        print('end tuning...')

        best_params = study.best_params
        best_score = study.best_value
        best_params = {f'lgbm__{k}':v for k,v in best_params.items()}
        # self.model.set_params(**best_params)
        print(f'best params: {best_params}')
        print(f'best score: {best_score}')
        return best_params
    
    def build(self, df):
        print('start build model...')
        df_train, df_test = train_test_split(df, random_state=42, test_size=self.cfg['test_size'],
                                            shuffle=True, stratify=df[self.target_col])
        
        test = split_dataset_by_target(df_train, self.target_col, self.cfg['n_split_df'])
        
        models = []
        
        for df in test:
            df_train, df_val = train_test_split(df, random_state=42, test_size=self.cfg['test_size'],
                                            shuffle=True, stratify=df[self.target_col])
            
            features = self.select_features(df_train, df_val)
            model = deepcopy(self.model)
            model.set_params(**features)
            best_params = self.params_tuning(df_train, df_val, model)
            model.set_params(**best_params)

            model.fit(df_train, df_train[self.target_col])

            y_pred = model.predict(df_val)
            metrics = roc_auc_score(df_val[self.target_col], y_pred)
            print(f'roc_auc on validate data = {metrics}')
            models.append(model)
        preds = []
        for m in models:
            preds.append(m.predict_proba(df_test)[:, 1])
        result = pd.DataFrame(preds).mean()
        print('metrics on test data:')
        evaluate_metrics(df_test['target'], result)
        return models
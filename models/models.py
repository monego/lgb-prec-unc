from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (train_test_split,
                                     RepeatedKFold,
                                     cross_val_score)
from sklearn.preprocessing import MinMaxScaler
import optuna
import pandas as pd
import lightgbm as lgb
import numpy as np
from lightgbm.basic import LightGBMError
from optuna.samplers import TPESampler


class PrecModel:

    def __init__(self, data, split_year, test_size, name):
        self.data = data
        self.name = name

        # Training data is all data before year 2018
        training_data = self.data.loc[data['year'] < 2018]

        self.X_training = training_data.drop(['year', 'prec_GPCP_roll'],
                                             axis=1)
        self.y_training = training_data[["prec_GPCP_roll"]]

        # Split training and validation sets, 75% and 25% in the article
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_training, self.y_training,
            test_size=test_size, random_state=42)

        # Test data is data where year >= 2018
        self.test_data = self.data.loc[data['year'] >= 2018]
        self.X_test = self.test_data.drop(['year', 'prec_GPCP_roll'], axis=1)
        self.y_test = self.test_data[["prec_GPCP_roll"]]

        # NOTE: Decision trees do not require normalization. See
        # https://towardsdatascience.com/do-decision-trees-need-feature-scaling-97809eaa60c6
        # It doesn't hurt, either. It will be removed in a future version.
        self._scale_features()

    def _scale_features(self):

        self.scaler_trainX = MinMaxScaler()
        self.scaler_valX = MinMaxScaler()
        self.scaler_testX = MinMaxScaler()
        self.scaler_trainy = MinMaxScaler()
        self.scaler_valy = MinMaxScaler()
        self.scaler_testy = MinMaxScaler()

        self.scaler_train_X = self.scaler_trainX.fit(self.X_train)
        self.scaler_val_X = self.scaler_valX.fit(self.X_val)
        self.scaler_test_X = self.scaler_testX.fit(self.X_test)
        self.scaler_train_y = self.scaler_trainy.fit(self.y_train)
        self.scaler_val_y = self.scaler_valy.fit(self.y_val)
        self.scaler_test_y = self.scaler_testy.fit(self.y_training)

        self.X_train = pd.DataFrame(self.scaler_train_X.transform(self.X_train),
                                    columns=self.X_train.columns,
                                    index=self.X_train.index)
        self.X_val = pd.DataFrame(self.scaler_val_X.transform(self.X_val),
                                  columns=self.X_val.columns,
                                  index=self.X_val.index)
        self.X_test = pd.DataFrame(self.scaler_test_X.transform(self.X_test),
                                   columns=self.X_test.columns,
                                   index=self.X_test.index)

        self.y_train = pd.DataFrame(self.scaler_train_y.transform(self.y_train),
                                    columns=self.y_train.columns,
                                    index=self.y_train.index)
        self.y_val = pd.DataFrame(self.scaler_val_y.transform(self.y_val),
                                  columns=self.y_val.columns,
                                  index=self.y_val.index)
        self.y_test = pd.DataFrame(self.scaler_test_y.transform(self.y_test),
                                   columns=self.y_test.columns,
                                   index=self.y_test.index)

        self.DX = lgb.Dataset(self.X_train, self.y_train)
        self.Dval = lgb.Dataset(self.X_val, self.y_val)
        self.Dtest = lgb.Dataset(self.X_test, self.y_test)

    def _objective(self, trial):
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-2, 0.1),
            "num_leaves": trial.suggest_int(
                "num_leaves", 20, 256),
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 1000),
            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-2, 1.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-2, 1.0, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 0.7, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 1000),
        }

        gbm = lgb.LGBMRegressor(**params,
                                seed=42)

        # Create a 5-fold cross validation scheme
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)

        n_scores = cross_val_score(gbm, self.X_train, self.y_train,
                                   scoring='neg_mean_squared_error',
                                   cv=cv, n_jobs=-1, error_score='raise',
                                   fit_params={'eval_metric': 'l2',
                                               'eval_set': (self.X_val,
                                                            self.y_val)})

        # Return the mean of scores, maximize for the best mean
        return np.mean(n_scores)

    def fit(self):

        study_name = 'lgb-{}'.format(self.name)
        storage_name = "sqlite:///lgb-{}.db".format(self.name)

        sampler = TPESampler(seed=42)

        self.study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            study_name=study_name,
            storage=storage_name,
            sampler=sampler,
            load_if_exists=True)

        self.evals = {}

        try:
            # Check whether a trained model exists and use that instead
            self.bst = lgb.Booster(
                model_file=f'models/lgb_{self.name}_model.txt')
            print("A saved model was found in disk. Using it instead...")

        except LightGBMError:

            self.study.optimize(self._objective, n_trials=100)

            self.bst = lgb.train(self.study.best_params,
                                 train_set=self.DX,
                                 valid_sets=[self.Dval],
                                 callbacks=[lgb.record_evaluation(self.evals)])

    def plot_learning_curve(self):
        lgb.plot_metric(self.evals)

    def calc_variance(self):
        """ Calculate the rolling variance, every 3 months,
        of the prediction error to measure uncertainty. """

        # Perform predictions and then unscale data
        preds_X_train = self.bst.predict(self.X_train).reshape(-1, 1)
        preds_X_train_denorm = pd.DataFrame({'prec_lgb': self.scaler_train_y.inverse_transform(preds_X_train).ravel()}, index=self.X_train.index)

        preds_X_val = self.bst.predict(self.X_val).reshape(-1, 1)
        preds_X_val_denorm = pd.DataFrame({'prec_lgb': self.scaler_val_y.inverse_transform(preds_X_val).ravel()}, index=self.X_val.index)

        preds_X_test = self.bst.predict(self.X_test).reshape(-1, 1)
        preds_X_test_denorm = pd.DataFrame({'prec_lgb': self.scaler_test_y.inverse_transform(preds_X_test).ravel()}, index=self.X_test.index)

        # Concatenate training, validation and test sets
        prec_lgb = pd.concat([preds_X_train_denorm,
                              preds_X_val_denorm,
                              preds_X_test_denorm], axis=0)
        prec_lgb.sort_index(inplace=True)

        # Calculate the error between observed data and predictions
        prec_y = self.data[["prec_GPCP_roll"]]

        error = prec_lgb['prec_lgb'] - prec_y['prec_GPCP_roll']

        error_months = []

        for m in range(12):
            unc = error[m::12].rolling(3, center=True).var()
            unc.iloc[0] = np.median(unc.dropna())
            unc.iloc[-1] = np.median(unc.dropna())
            error_months.append(unc)

        self.data = self.data.assign(prec_lgb=prec_lgb)
        self.data = self.data.assign(unc=pd.concat(error_months).sort_index())

    def train_test_error(self):
        preds_train = self.bst.predict(self.X_train)
        preds_test = self.bst.predict(self.X_test)
        print("Training error: {}".format(mean_squared_error(preds_train,
                                                             self.y_train)))
        print("Test error: {}".format(mean_squared_error(preds_test,
                                                         self.y_test)))

    def print_errors(self):
        """ Display prediction errors for each month """
        preds_X_test = self.bst.predict(self.X_test).reshape(-1, 1)
        preds_X_test_denorm = pd.DataFrame({'prediction': self.scaler_test_y.inverse_transform(preds_X_test).ravel()}, index=self.X_test.index)
        error_df = pd.DataFrame(
            {'year': self.test_data['year'],
             'month': self.test_data['month'],
             'reference': self.test_data['prec_GPCP_roll'],
             'prediction': preds_X_test_denorm['prediction']})
        for year in [2018, 2019]:
            for month in range(1, 13):
                qry = error_df.query(f'year == {year} and month == {month}')
                me = np.mean(qry['prediction'] - qry['reference'])
                rmse = np.sqrt(mean_squared_error(qry['prediction'],
                                                  qry['reference']))
                print(f"ME {year}/{month}: {me}")
                print(f"RMSE {year}/{month}: {rmse}")
                print("")

    def best_params(self):
        print(self.study.best_params)

    def save_data(self, fname):
        self.data.to_excel(fname, index=False)

    def save_trials(self):
        df = self.study.trials_dataframe()
        df.to_excel("best-params-lgb-{}.xlsx".format(self.name))

    def save_model(self):
        self.bst.save_model('lgb_{}_model.txt'.format(self.name),
                            num_iteration=self.bst.best_iteration)


class UncModel(PrecModel):

    def __init__(self, data, split_year, test_size, name):
        self.data = data
        self.name = name
        training_data = self.data.loc[data['year'] < 2018]
        self.X_training = training_data.drop(['year', 'unc'], axis=1)
        self.y_training = training_data[["unc"]]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_training, self.y_training,
            test_size=test_size, random_state=42)

        self.test_data = data.loc[data['year'] >= 2018]
        self.X_test = self.test_data.drop(['year', 'unc'], axis=1)
        self.y_test = self.test_data[["unc"]]

        self._scale_features()

    def print_errors(self):
        """ Display prediction errors for each month """
        preds_X_test = self.bst.predict(self.X_test).reshape(-1, 1)
        preds_X_test_denorm = pd.DataFrame({'prediction': self.scaler_test_y.inverse_transform(preds_X_test).ravel()}, index=self.X_test.index)

        error_df = pd.DataFrame(
            {'year': self.test_data['year'],
             'month': self.test_data['month'],
             'reference': self.test_data['unc'],
             'prediction': preds_X_test_denorm['prediction']})
        for year in [2018, 2019]:
            for month in range(1, 13):
                qry = error_df.query(f'year == {year} and month == {month}')
                me = np.mean(qry['prediction'] - qry['reference'])
                rmse = np.sqrt(mean_squared_error(qry['prediction'],
                                                  qry['reference']))
                print(f"ME {year}/{month}: {me}")
                print(f"RMSE {year}/{month}: {rmse}")
                print("")

    def save_data(self):
        preds_X_train = self.bst.predict(self.X_train).reshape(-1, 1)
        preds_X_train_denorm = pd.DataFrame({'unc_lgb': self.scaler_train_y.inverse_transform(preds_X_train).ravel()}, index=self.X_train.index)

        preds_X_val = self.bst.predict(self.X_val).reshape(-1, 1)
        preds_X_val_denorm = pd.DataFrame({'unc_lgb': self.scaler_val_y.inverse_transform(preds_X_val).ravel()}, index=self.X_val.index)

        preds_X_test = self.bst.predict(self.X_test).reshape(-1, 1)
        preds_X_test_denorm = pd.DataFrame({'unc_lgb': self.scaler_test_y.inverse_transform(preds_X_test).ravel()}, index=self.X_test.index)

        unc_lgb = pd.concat([preds_X_train_denorm,
                             preds_X_val_denorm,
                             preds_X_test_denorm], axis=0)
        unc_lgb.sort_index(inplace=True)

        self.data = self.data.assign(unc_lgb=unc_lgb)

        self.data.to_excel('uncertainty-estimation.xlsx')

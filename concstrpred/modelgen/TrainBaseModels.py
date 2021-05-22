# Import pandas library
import pandas as pd

# Import pickle library
import pickle as pkl

# Import Train-Test split library
from sklearn.linear_model import LinearRegression

# Import RandomForestRegressor library
from sklearn.ensemble import RandomForestRegressor

# Import XGBRegressor library
from xgboost import XGBRegressor

# Import LightGBM library
import lightgbm as lgb

# Import GradientBoostingRegressor library
from sklearn.ensemble import GradientBoostingRegressor

# Import Train-Test split library
from sklearn.model_selection import train_test_split

# Import MSE and MAE modules from metrics computing library
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import warnings
import warnings
warnings.filterwarnings('ignore')

# Import project specific libraries
from ProjectConstants import SPLIT_RATIO, RANDOM_STATE

"""
This class loads the final dataset, splits them into train-test, trains base models, and stores them.
"""

class TrainBaseModels():

    # init method or constructor
    def __init__(self):
        self.__load_source_data()

    def __load_source_data(self):
        self.df = pd.read_csv("data/output/Concrete_final.csv")

    def split_data(self):
        # Split the dataset
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.df.drop('compressive_strength', axis = 1),
                                self.df['compressive_strength'], test_size=SPLIT_RATIO, random_state=RANDOM_STATE)

    def train_linear_regression(self):
        # Generate a Linear Regression object
        lr_model = LinearRegression()

        # Train a Linear Regression model with Train dataset
        lr_model.fit(self.X_train, self.Y_train)

        # To persist the base model into hard-disk, uncomment the below line
        # pkl.dump(lr_model, open("model/lr_base_model.mdl", 'wb'))

        # Predict the outcome
        y_hat_lr = lr_model.predict(self.X_test)

        # Compute the RMSE score
        rmse_lr = round(mean_squared_error(self.Y_test, y_hat_lr, squared=False), 3)   # For MSE, squared=True

        # Print RMSE score
        print("RMSE of Linear Regression model:", rmse_lr)

        return rmse_lr

    def train_random_forest(self):
        # Generate a Random Forest Regressor object
        rf_model = RandomForestRegressor(random_state=RANDOM_STATE)

        # Train a Random Forest model with Train dataset
        rf_model.fit(self.X_train, self.Y_train)

        # To persist the base model into hard-disk, uncomment the below line
        # pkl.dump(rf_model, open("model/rf_base_model.mdl", 'wb'))

        # Predict the outcome
        y_hat_rf = rf_model.predict(self.X_test)

        # Compute the RMSE score
        rmse_rf = round(mean_squared_error(self.Y_test, y_hat_rf, squared=False), 3)

        # Print RMSE score
        print("RMSE of Random Forest model:", rmse_rf)

        return rmse_rf

    def train_xgboost(self):
        # Generate a XGBoost object
        xgb_model = XGBRegressor()

        # Train a XGBoost model with Train dataset
        xgb_model.fit(self.X_train, self.Y_train)

        # To persist the base model into hard-disk, uncomment the below line
        # pkl.dump(xgb_model, open("model/xgb_base_model.mdl", 'wb'))

        # Predict the outcome
        y_hat_xgb = xgb_model.predict(self.X_test)

        # Compute the RMSE score
        rmse_xgb = round(mean_squared_error(self.Y_test, y_hat_xgb, squared=False), 3)

        # Print RMSE score
        print("RMSE of XGBoost model:", rmse_xgb)

        return rmse_xgb

    def train_lightgbm(self):
        # Prepare train data object specific to LGBM
        d_train = lgb.Dataset(self.X_train, label=self.Y_train)

        # Train a LGBM model with Train object
        lgb_model = lgb.train(train_set=d_train, params={})

        # To persist the base model into hard-disk, uncomment the below line
        # pkl.dump(lgb_model, open("model/lgb_base_model.mdl", 'wb'))

        # Predict the outcome
        y_hat_lgb = lgb_model.predict(self.X_test)

        # Compute the RMSE score
        rmse_lgb = round(mean_squared_error(self.Y_test, y_hat_lgb, squared=False), 3)

        # Print RMSE score
        print("RMSE of LightGBM model:", rmse_lgb)

        return rmse_lgb

    def train_gbm(self):
        # Generate a Gradient Boosting Regressor object
        gbm_model = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=1, random_state=RANDOM_STATE)

        # Train a  Gradient Boosting Regressor model with Train dataset
        gbm_model.fit(self.X_train, self.Y_train)

        # To persist the base model into hard-disk, uncomment the below line
        # pkl.dump(gbm_model, open("model/gbm_base_model.mdl", 'wb'))

        # Predict the outcome
        y_hat_gbm = gbm_model.predict(self.X_test)

        # Compute the RMSE score
        rmse_gbm = round(mean_squared_error(self.Y_test, y_hat_gbm, squared=False), 3)

        # Print RMSE score
        print("RMSE of Gradient Boosting Regressor model:", rmse_gbm)

        return rmse_gbm

# Class ends here

def export_performance_metrics(rmse_lr, rmse_rf, rmse_xgb, rmse_lgb, rmse_gbm):
    perf_metrics_df = pd.DataFrame({'Linear Regression': [rmse_lr],
                                    'Random Forest': [rmse_rf],
                                    'XGBoost': [rmse_xgb],
                                    'LightGBM': [rmse_lgb],
                                    'GBM': [rmse_gbm]})

    perf_metrics_df.to_csv("data/output/baseline_metrics.csv", index=False)

def main():
    base_models = TrainBaseModels()
    base_models.split_data()
    rmse_lr = base_models.train_linear_regression()
    rmse_rf = base_models.train_random_forest()
    rmse_xgb = base_models.train_xgboost()
    rmse_lgb = base_models.train_lightgbm()
    rmse_gbm = base_models.train_gbm()

    export_performance_metrics(rmse_lr, rmse_rf, rmse_xgb, rmse_lgb, rmse_gbm)

if __name__ == "__main__":
    main()

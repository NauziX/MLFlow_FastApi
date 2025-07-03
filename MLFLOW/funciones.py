import pandas as pd
import numpy as np
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


RND = np.random.RandomState(42)

"Funcion para la carga de datos"


def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df


# Funciones para el visualizado de datos"


def show_histogram(df):
    for column in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


def show_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()


# Funciones para el tratamiento de datos"


def split_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND)
    return X_train, X_test, y_train, y_test


# Funciones para el seguimiento de experimentos con MLflow"


def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, n_estimators):

    mlflow.set_experiment(nombre_job)

    for n in n_estimators:
        with mlflow.start_run(run_name=f"rf_{n}"):
            rf = RandomForestRegressor(
                n_estimators=n,
                min_samples_leaf=2,
                random_state=RND
            )

            rf.fit(x_train, y_train)

            y_pred_train = rf.predict(x_train)
            y_pred_test = rf.predict(x_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test,  y_pred_test)

            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_test, y_pred_test)
            mlflow.log_params({"n_estimators": n, "min_samples_leaf": 2})
            mlflow.log_metrics({
                "rmse_train": rmse_train,
                "rmse_test":  rmse_test,
                "r2_test":    r2_test
            })

            mlflow.sklearn.log_model(rf, artifact_path="rf_model")

    print("Entrenamiento RandomForestRegressor terminado correctamente")


def mlflow_tracking_xgboost(nombre_job, x_train, x_test, y_train, y_test, n_estimators):
    mlflow.set_experiment(nombre_job)

    for n in n_estimators:
        with mlflow.start_run(run_name=f"xgb_{n}"):
            xgb = XGBRegressor(
                n_estimators=n,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=RND,
                n_jobs=-1
            )

            xgb.fit(x_train, y_train)

            y_pred_train = xgb.predict(x_train)
            y_pred_test = xgb.predict(x_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)

            mlflow.log_param('n_estimators', n)
            mlflow.log_metrics({
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_test': r2_test
            })

            mlflow.sklearn.log_model(xgb, artifact_path='xgb_model')

    print("Entrenamiento XGBoost terminado correctamente")


def mlflow_tracking_knn(nombre_job, x_train, x_test, y_train, y_test, k_list):
    mlflow.set_experiment(nombre_job)

    for k in k_list:
        with mlflow.start_run(run_name=f"knn_{k}"):
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(x_train, y_train)

            y_pred_train = knn.predict(x_train)
            y_pred_test = knn.predict(x_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)

            mlflow.log_param('n_neighbors', k)
            mlflow.log_metrics({
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_test': r2_test
            })

            mlflow.sklearn.log_model(knn, artifact_path='knn_model')

    print("Entrenamiento K-NN terminado correctamente")

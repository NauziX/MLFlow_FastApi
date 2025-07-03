
import funciones as func


def main():

    df = func.load_data()
    x_train, x_test, y_train, y_test = func.split_data(df)
    func.mlflow_tracking('RFR-v5', x_train, x_test, y_train, y_test,
                         n_estimators=[50, 100])
    func.mlflow_tracking_xgboost('XGB-v5',
                                 x_train, x_test, y_train, y_test,
                                 n_estimators=[100, 300])
    func.mlflow_tracking_knn('KNN-v5',
                             x_train, x_test, y_train, y_test,
                             k_list=[3, 5, 7])


if __name__ == '__main__':
    main()

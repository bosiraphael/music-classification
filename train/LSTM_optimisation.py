import time
import optuna
from data_loader import prepare_datasets
from train_LSTM import train_LSTM

if __name__ == "__main__":
    X_train, X_test, X_validation, y_train, y_test, y_validation = prepare_datasets(0.2, 0.1)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        return train_LSTM(X_train, X_test, X_validation, y_train, y_test, y_validation, batch_size, 200, lr, dropout)[2][-1]

    t0 = time.time()
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    t1 = time.time()

    with open('trained_models/parameters/LSTM_parameters.txt', 'w') as f:
        f.write(f'time : {t1-t0}\nbest params : {study.best_params}\nbest value : {study.best_value}\nbest trial : {study.best_trial}\n\n{study.trials}')
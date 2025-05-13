import h5py
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV  # Requires scikit-optimize to be installed
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import os

# Install required packages (run in terminal)
# pip install scikit-optimize

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month in months:
    # Data loading (unchanged)
    with h5py.File(r"E:\Quasi-SMAP\04_gap_filing\01_features" + '/' + month + ".h5", 'r') as hf:
        factors = hf['factors'][:]
        cci = hf['cci'][:]
        print(cci)

    x_train, x_test, y_train, y_test = train_test_split(factors, cci,
                                                        train_size=0.8, random_state=42)

    # Parameter space for Bayesian optimization (supports continuous range)
    param_space = {
        'max_depth': Integer(6, 12),  # Integer range
        'eta': Real(0.05, 0.3, prior='log-uniform'),  # Continuous (log scale)
        'subsample': Real(0.6, 0.8),
        'colsample_bytree': Real(0.6, 0.8),
        'min_child_weight': Integer(7, 20),
        'gamma': Real(0.1, 1),
        'alpha': Real(0.1, 5),
        'lambda': Real(1, 5)  # Change to 'reg_lambda' if needed
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                 tree_method="hist",
                                 device="cuda",
                                 eval_metric=['rmse']  # Evaluation metric
                                 )

    # Bayesian search configuration
    bayes_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=param_space,
        n_iter=30,  # Bayesian search typically needs fewer iterations than random search
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=1  # Keep this for GPU compatibility
    )

    # Training (unchanged)
    bayes_search.fit(x_train, y_train)

    # The following code for output and saving is unchanged
    print("Best parameters found: ", bayes_search.best_params_)
    best_model = bayes_search.best_estimator_

    # Train the final model with the best parameters
    preds_train = best_model.predict(x_train)
    preds_test = best_model.predict(x_test)

    # Calculate evaluation metrics
    score1 = r2_score(preds_train, y_train)
    bias1 = np.mean(preds_train) - np.mean(y_train)
    rmse1 = np.sqrt(mean_squared_error(preds_train, y_train))
    ubrmse1 = np.sqrt(rmse1 ** 2 - bias1 ** 2)
    score2 = r2_score(preds_test, y_test)
    bias2 = np.mean(preds_test) - np.mean(y_test)
    rmse2 = np.sqrt(mean_squared_error(preds_test, y_test))
    ubrmse2 = np.sqrt(rmse2 ** 2 - bias2 ** 2)

    print('Training Set R2     is  : {}'.format('%.6f' % score1))
    print('Training Set RMSE   is  : {}'.format('%.6f' % rmse1))
    print('Training Set BIAS   is  : {}'.format('%.6f' % bias1))
    print('Training Set ubRMSE is  : {}'.format('%.6f' % ubrmse1))
    print('Testing  Set R2     is  : {}'.format('%.6f' % score2))
    print('Testing  Set RMSE   is  : {}'.format('%.6f' % rmse2))
    print('Testing  Set BIAS   is  : {}'.format('%.6f' % bias2))
    print('Testing  Set ubRMSE is  : {}'.format('%.6f' % ubrmse2))

    models_path = r"E:\Quasi-SMAP\04_gap_filing\02——model"
    best_model.save_model(os.path.join(models_path, month + '-1' + '.model'))
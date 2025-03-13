import Data_Editing_Helpers as DEH
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool

def decisiontreeRegressor(X_train, y_train):
    print("\nDecision Tree Regressor Started")
    dt_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor(random_state=301))
    ])
    param_grid = {
        'regressor__max_depth': [1, 3, 5, 7, 10],
        'regressor__min_samples_split': [2, 5, 10, 15, 20],
        'regressor__min_samples_leaf': [1, 3, 5, 10, 15, 20],
        'regressor__max_leaf_nodes': [None, 5, 10, 15, 20, 25],
        'regressor__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='r2', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_dt = random_search.best_estimator_

    print(f'Best parameters Decisiontree: {best_params}')
    DEH.saveModel(best_model_dt, './TrainedModels/decisiontreeRegressor.pkl')
    print("Decision Tree Regressor Finished")
    return best_model_dt

def linearRegressor(X_train, y_train):
    print("\nLinear Regressor Started")
    lr_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    lr_model.fit(X_train, y_train)
    DEH.saveModel(lr_model, './TrainedModels/linearRegressor.pkl')
    print("Linear Regressor Finished")
    return lr_model

def ridgeRegressor(X_train, y_train):
    print("\nRidge Regressor Started")
    ridge_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])
    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0, 100, 1000, 10000, 100000]
    }

    grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model_ridge = grid_search.best_estimator_

    print(f'Best parameters Ridge: {best_params}')
    DEH.saveModel(best_model_ridge, './TrainedModels/ridgeRegressor.pkl')
    print("Ridge Regressor Finished")
    return best_model_ridge

def lassoRegressor(X_train, y_train):
    print("\nLasso Regressor Started")
    lasso_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso())
    ])
    param_grid = {
        'regressor__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 1000, 10000]
    }

    grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model_lasso = grid_search.best_estimator_

    print(f'Best parameters Lasso: {best_params}')
    DEH.saveModel(best_model_lasso, './TrainedModels/lassoRegressor.pkl')
    print("Lasso Regressor Finished")
    return best_model_lasso

def randomForestRegressor(X_train, y_train):
    print("\nRandom Forest Regressor Started")
    rf_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=301))
    ])
    param_grid = {
        'regressor__n_estimators': [50, 100, 150, 300, 600],
        'regressor__max_depth': [None, 1, 2, 3, 5],
        'regressor__min_samples_split': [2, 5, 10, 20],
        'regressor__min_samples_leaf': [1, 5, 10, 20, 40, 60, 80, 100],
    }

    random_search = RandomizedSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_rf = random_search.best_estimator_

    print(f'Best parameters RandomForest: {best_params}')
    DEH.saveModel(best_model_rf, './TrainedModels/randomForestRegressor.pkl')
    print("Random Forest Regressor Finished")
    return best_model_rf

def gradientBoostingRegressor(X_train, y_train):
    print("\nGradient Boosting Regressor Started")
    gb_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(random_state=301))
    ])
    param_grid = {
        'regressor__n_estimators': [50, 100, 150, 300, 600, 900],
        'regressor__learning_rate': [0.001, 0.01, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5, 10, 20],
        'regressor__min_samples_leaf': [1, 5, 10, 20],
    }

    random_search = RandomizedSearchCV(gb_model, param_grid, cv=5, scoring='r2', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_gb = random_search.best_estimator_

    print(f'Best parameters GradientBoosting: {best_params}')
    DEH.saveModel(best_model_gb, './TrainedModels/gradientBoostingRegressor.pkl')
    print("Gradient Boosting Regressor Finished")
    return best_model_gb

def catBoostRegressor(X_train, y_train):
    print("\nCatBoost Regressor Started")

    # Create a pipeline with normalization and the CatBoost regressor
    cat_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', CatBoostRegressor(random_state=301, verbose=0))
    ])

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'regressor__iterations': [100, 200, 300, 400, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__depth': [2, 3, 5, 7, 9, 12, 15],
        'regressor__l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    # Perform RandomizedSearchCV to find the best parameters
    random_search = RandomizedSearchCV(cat_model, param_grid, cv=5, scoring='r2', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    # Get the best parameters and the best model
    best_params = random_search.best_params_
    best_model_cat = random_search.best_estimator_

    # Evaluate the model on the testing data
    test_score = best_model_cat.score(X_train, y_train)

    print(f'Best parameters CatBoost: {best_params}')
    print(f'CatBoost Test R^2 Score: {test_score}')
    DEH.saveModel(best_model_cat, './TrainedModels/catBoostRegressor.pkl')
    print("CatBoost Regressor Finished")
    return best_model_cat
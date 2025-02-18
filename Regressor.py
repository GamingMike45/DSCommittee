import Data_Editing_Helpers as DEH
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def decisiontreeRegressor(X_train, y_train):
    print("\nDecision Tree Regressor Started")
    dt_model = DecisionTreeRegressor(random_state=301)
    param_grid = {
        'max_depth': [1, 3, 5, 7, 9, 15, 30], #Max depth of the tree
        'min_samples_split': [2, 5, 10], #Min sample 
        'min_samples_leaf': [1, 5, 10], # Min samples in a leaf
        'max_features': [None, 'auto', 'sqrt, log2'], # Number of features per split
        'criterion': ['squared_error', 'Friendman_mse', 'poisson'], # Function to measure quality of a split
        'splitter': ['best', 'random'], # Strategy to choose split
        'max_leaf_nodes': [None, 5, 10, 15], # Max number of leaf nodes
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2] # Min weight fraction of the sum total of weights

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
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    DEH.saveModel(lr_model, './TrainedModels/linearRegressor.pkl')
    print("Linear Regressor Finished")
    return lr_model

def ridgeRegressor(X_train, y_train):
    print("\nRidge Regressor Started")
    ridge_model = Ridge()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0]
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
    lasso_model = Lasso()
    param_grid = {
        'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 1000, 1000]
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
    rf_model = RandomForestRegressor(random_state=301)
    param_grid = {
        'n_estimators': [50, 100, 150],  # Number of trees
        'max_depth': [0, 1, 2, 3, 5, 7, 9], #Depth of each tree
        'min_samples_split': [2, 5, 10], # Min sample split to a node
        'min_samples_leaf': [1, 5, 10], # Min samples in a leaf
        'max_features': ['auto', 'sqrt, log2'], # Number of features per split
        'bootstrap': [True, False] #sample Method
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
    gb_model = GradientBoostingRegressor(random_state=301)
    param_grid = {
        'n_estimators': [50, 100, 150, 300, 800], #Number of boosting stageds
        'learning_rate': [0.001, 0.01, 0.1, 0.5], # Shrinks the contributuon of trees
        'max_depth': [3, 5, 7, 9], #Tree depth
        'min_samples_split': [2, 5, 10], # Min sample split to a node
        'min_samples_leaf': [1, 5, 10], # Min samples in a leaf
        'max_features': ['auto', 'sqrt, log2'], # Number of features per split
        'subsample': [0.5, 0.7, 1.0] # Fraction of samples to be used for fitting
    }
    random_search = RandomizedSearchCV(gb_model, param_grid, cv=5, scoring='r2', n_iter=10, n_iter_no_change=10,random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_gb = random_search.best_estimator_

    print(f'Best parameters GradientBoosting: {best_params}')
    DEH.saveModel(best_model_gb, './TrainedModels/gradientBoostingRegressor.pkl')
    print("Gradient Boosting Regressor Finished")
    return best_model_gb
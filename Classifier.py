import Data_Editing_Helpers as DEH
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def decisiontreeClassifier(X_train, y_train):
    print("\nDecision Tree Classifier Started")
    dt_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=301))
    ])
    param_grid = {
        'classifier__max_depth': [1, 3, 5, 7, 10],
        'classifier__min_samples_split': [2, 5, 10, 15, 20],
        'classifier__min_samples_leaf': [1, 3, 5, 10, 15, 20],
        'classifier__max_leaf_nodes': [None, 5, 10, 15, 20, 25],
        'classifier__min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_dt = random_search.best_estimator_

    print(f'Best parameters Decision Tree: {best_params}')
    DEH.saveModel(best_model_dt, './TrainedModels/decisiontreeClassifier.pkl')
    print("Decision Tree Classifier Finished")
    return best_model_dt

def knnClassifier(X_train, y_train):
    print("\nKNN Classifier Started")
    knn_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])
    param_grid = {
        'classifier__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'classifier__leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__p': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    random_search = RandomizedSearchCV(knn_model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_knn = random_search.best_estimator_

    print(f'Best parameters KNN: {best_params}')
    DEH.saveModel(best_model_knn, './TrainedModels/knnClassifier.pkl')
    print("KNN Classifier Finished")
    return best_model_knn

def adaboostClassifier(X_train, y_train):
    print("\nAdaBoost Classifier Started")
    dt_model = AdaBoostClassifier()

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
    }
    random_search = GridSearchCV(dt_model, param_grid)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_ada = random_search.best_estimator_

    print(f'Best parameters: {best_params}')

    DEH.saveModel(best_model_ada, './TrainedModels/adaModel.pkl')
    print("AdaBoost Classifier Finished")
    return best_model_ada
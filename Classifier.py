import Data_Editing_Helpers as DEH
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV




def decisiontreeClassifier(X_train, y_train):
    print("\nDecision Tree Classifier Started")
    dt_model = DecisionTreeClassifier(random_state=1235)
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }
    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='r2', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_dt = random_search.best_estimator_

    print(f'Best parameters Decisiontree: {best_params}')
    DEH.saveModel(best_model_dt, './TrainedModels/decisiontreeclassifier.pkl')
    print("Decision Tree Classifier Finished")
    return best_model_dt


def knnClassifier(X_train, y_train):
    print("\nKNN Classifier Started")
    dt_model = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "leaf_size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "p": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)


    best_params = random_search.best_params_
    best_model_knn = random_search.best_estimator_

    print(f'Best parameters KNN: {best_params}')
    DEH.saveModel(best_model_knn, './TrainedModels/knnModel.pkl')
    print("KNN Classifier Finished")
    return best_model_knn

def adaboostClassifier(X_train, y_train):
    print("\nAdaBoost Classifier Started")
    dt_model = AdaBoostClassifier()

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME']
    }
    random_search = GridSearchCV(dt_model, param_grid)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_ada = random_search.best_estimator_

    print(f'Best parameters: {best_params}')

    DEH.saveModel(best_model_ada, './TrainedModels/adaModel.pkl')
    print("AdaBoost Classifier Finished")
    return best_model_ada
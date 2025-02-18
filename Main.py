import Data_Editing_Helpers as DEH
import Classifier as CLF
import Regressor as RGS
import threading
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

## Loading ##
test = pd.read_csv("Data/test.csv")
train = pd.read_csv("Data/train.csv")

y_name = 'Price' # What you're trying to predict
x_name = 'id' # User id. Drop this column

# Set this to True if you want to run regression models, False for classification models
is_regression = True

## Wrangling ##
#Future implementation will remove map_seasons and convert_strings_to_ascii
train, test = DEH.map_seasons(train, test)
#train = DEH.convert_strings_to_ascii(train)
#test = DEH.convert_strings_to_ascii(test)
train, test = DEH.dropUnusedColumns(train, test, y_name, x_name)
train = DEH.remove_blank_rows(train, y_name)

# Using 99 as a filler for NA's. Will change to use Random Forest for filling NA's 
train, test = DEH.fill_NA(train, test, fill=99)
X_train, X_test, y_train, y_test = DEH.traintestslpit(train, y_name)

## Visualizing ##
# DEH.makeSNS(train)   # This outputs all graphs, can be annoying

## Function to evaluate and store model performance ##
def evaluate_model(model_func, X_train, y_train, X_test, y_test, model_name, results, is_regression):
    model = model_func(X_train, y_train)
    predictions = model.predict(X_test)
    if is_regression:
        score = r2_score(y_test, predictions) * 100
    else:
        score = accuracy_score(y_test, predictions) * 100
    results.append({"model": model_name, "score": score, "model_obj": model})
    print(f"{model_name} score: {score:.3f}")

## Training Models ##
results = []
# List of models to evaluate
t = []

if is_regression:
    # Evaluate regression models
    t1 = threading.Thread(target=evaluate_model(RGS.decisiontreeRegressor, X_train, y_train, X_test, y_test, "Decision Tree Regressor", results, is_regression))
    t2 = threading.Thread(target=evaluate_model(RGS.linearRegressor, X_train, y_train, X_test, y_test, "Linear Regressor", results, is_regression))
    t3 = threading.Thread(target=evaluate_model(RGS.ridgeRegressor, X_train, y_train, X_test, y_test, "Ridge Regressor", results, is_regression))
    t4 = threading.Thread(target=evaluate_model(RGS.lassoRegressor, X_train, y_train, X_test, y_test, "Lasso Regressor", results, is_regression))
    t5 = threading.Thread(target=evaluate_model(RGS.randomForestRegressor, X_train, y_train, X_test, y_test, "Random Forest Regressor", results, is_regression))
    t6 = threading.Thread(target=evaluate_model(RGS.gradientBoostingRegressor, X_train, y_train, X_test, y_test, "Gradient Boosting Regressor", results, is_regression))
    t.extend([t1, t2, t3, t4])  #Add to list as models increase
    for i in t:
        i.start()
else:
    # Evaluate classification models
    t1 = threading.Thread(target=evaluate_model(CLF.decisiontreeClassifier, X_train, y_train, X_test, y_test, "Decision Tree Classifier", results, is_regression))
    t2 = threading.Thread(target=evaluate_model(CLF.knnClassifier, X_train, y_train, X_test, y_test, "KNN Classifier", results, is_regression))
    t3 = threading.Thread(target=evaluate_model(CLF.adaboostClassifier, X_train, y_train, X_test, y_test, "AdaBoost Classifier", results, is_regression))
    t.extend([t1, t2, t3]) #Add to list as models increase
    for i in t:
        i.start()

for i in t: 
    i.join()

## Select the best model ##
best_model = max(results, key=lambda x: x["score"])
print(f"Best model: {best_model['model']} with score: {best_model['score']:.3f}")

## Making Predictions ##
predictions = best_model["model_obj"].predict(test)

## Submitting ##
DEH.generate_submission(predictions, x_name, y_name)

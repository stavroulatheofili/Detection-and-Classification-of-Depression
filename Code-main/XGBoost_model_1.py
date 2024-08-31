from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev
from balance_dataset import balance_dataset


# function and assign labels for second task
def assign_targets(analytic, handcraft_features):
    for i in range(0, len(analytic)):
        if analytic['target'][i] == 1:
            first_healthy = i
            break
    # assign the target column
    handcraft_features['target'] = 0
    for i in range(0, int(first_healthy / 48)):
        handcraft_features.loc[i, 'target'] = 0
    for i in range(int(first_healthy / 48), len(handcraft_features)):
        handcraft_features.loc[i, 'target'] = 1
    return handcraft_features


def create_handcraft_features(data):
    # mean value of activity for every 30 mins
    means = []

    # activity for every 30 min of a whole day
    max_activity = []
    min_activity = []
    stds = []

    # patient's id
    # p_id = []

    data2 = pd.DataFrame()

    activity_list = []
    for i in range(0, len(data)):
        activity_list.append(data.iloc[i]["activity"])
        if ((i - 1) % 48 == 0 and i != 0):
            means.append(mean(activity_list))
            stds.append(stdev(activity_list))
            max_activity.append(max(activity_list))
            min_activity.append(min(activity_list))
            # p_id.append(data.iloc[int(i / 48)]["patient"])
            activity_list = []

    data2["means"] = means
    data2["stds"] = stds
    data2["max"] = max_activity
    data2["min"] = min_activity
    # data2["patient"] = p_id

    return data2


analytic_train = pd.read_csv('Data/train1.csv')
analytic_test = pd.read_csv('Data/test1.csv')
analytic_validation = pd.read_csv("Data/validation1.csv")

df_train = assign_targets(analytic_train, create_handcraft_features(analytic_train))
df_test = assign_targets(analytic_test, create_handcraft_features(analytic_test))
df_validation = assign_targets(analytic_validation, create_handcraft_features(analytic_validation))

# balance the dataset
df_train = balance_dataset(df_train)
df_test = balance_dataset(df_test)
df_validation = balance_dataset(df_validation)

# split to features and targets
X_train = df_train.drop(['target'], axis=1)
y = df_train['target']
X_test = df_test.drop(['target'], axis=1)
y_test = df_test['target']
scaler = StandardScaler()
X = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Grid search
# params = {
#     'min_child_weight': [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 15],
#     'gamma': [0, 0.3, 0.5, 1, 1.3, 1.4, 1.5, 1.6, 1.7, 2, 5],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
#     'max_depth': [1, 2, 3, 4, 5, 6],
#     'learning_rate': [0.02, 0.01, 0.05, 0.1, 0.4, 0.5, 0.6]
# }
#
# param_comb = 10
# kfold = KFold(n_splits=10, shuffle=True, random_state=2023)
# xgb = xgboost.XGBClassifier(n_estimators=600, objective='binary:logistic',
#                             nthread=1)
# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4,
#                                    cv=kfold.split(X, y), verbose=3, random_state=2023)
#
# random_search.fit(X_train, y)
# print(random_search.best_score_)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)

model = xgboost.XGBClassifier(learning_rate=0.05, n_estimators=600, objective='binary:logistic',
                              nthread=1, max_depth=2, min_child_weight=2, gamma=0.3, colsample_bytree=1.0,
                              subsample=1, random_state=2023)
# kfold = KFold(n_splits=10, shuffle=True, random_state=2022)
# results = cross_val_score(model, X, y, cv=kfold)
model.fit(X, y)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# results = cross_val_score(model, X, y, cv=kfold)
# print("Accuracy: %.2f%%" % (results.mean() * 100))

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


def create_handcraft_features(data):
    # mean value of activity for every 30 mins
    means = []

    # activity for every 30 min of a whole day
    max_activity = []
    min_activity = []
    stds = []

    # patient's id
    p_id = []

    data2 = pd.DataFrame()
    personal_data = pd.read_csv('Data/scores_for classification_2.csv')

    activity_list = []
    for i in range(0, len(data)):
        activity_list.append(data.iloc[i]["activity"])
        if ((i - 1) % 48 == 0 and i != 0):
            means.append(mean(activity_list))
            stds.append(stdev(activity_list))
            max_activity.append(max(activity_list))
            min_activity.append(min(activity_list))
            p_id.append(data.iloc[i]["patient"])
            activity_list = []

    data2["means"] = means
    data2["stds"] = stds
    data2["max"] = max_activity
    data2["min"] = min_activity
    data2["patient"] = p_id

    age = []
    gender = []
    target = []
    inpatient = []
    edu = []
    marriage = []
    work = []

    for i in range(0, len(data2)):
        patient_id = int(data2.iloc[i]["patient"])
        age.append(personal_data.iloc[patient_id - 1]["age"])
        gender.append(personal_data.iloc[patient_id - 1]["gender"])
        target.append(personal_data.iloc[patient_id - 1]["afftype"]%2)
        inpatient.append(personal_data.iloc[patient_id - 1]["inpatient"])
        edu.append(personal_data.iloc[patient_id - 1]["edu"])
        marriage.append(personal_data.iloc[patient_id - 1]["marriage"])
        work.append(personal_data.iloc[patient_id - 1]["work"])

    data2["age"] = age
    data2["gender"] = gender
    data2["target"] = target
    data2["inpatient"] = inpatient
    data2["edu"] = edu
    data2["marriage"] = marriage
    data2["work"] = work

    return data2


analytic_train = pd.read_csv('Data/train2.csv')
analytic_test = pd.read_csv('Data/test2.csv')
analytic_validation = pd.read_csv("Data/validation2.csv")

df_train = create_handcraft_features(analytic_train)
df_test = create_handcraft_features(analytic_test)
df_validation = create_handcraft_features(analytic_validation)

# balance the dataset
df_train = balance_dataset(df_train)
df_test = balance_dataset(df_test)
df_validation = balance_dataset(df_validation)

# split to features and targets
X_train = df_train.drop(['target'], axis=1).drop(['patient'], axis=1)
y = df_train['target']
X_test = df_test.drop(['target'], axis=1).drop(['patient'], axis=1)
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
model.fit(X, y)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


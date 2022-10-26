from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

import sys
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
val_data_path = sys.argv[3]

train_data = pd.read_csv(train_data_path, sep=";")
val_data = pd.read_csv(val_data_path, sep=";")
test_data = pd.read_csv(test_data_path, sep=";")


def get_accuracy(model,features,labels):
    predictions = model.predict(features)
    c=0
    for idx,pred in enumerate(predictions):
        if(pred == labels[idx]):
            c+=1
    return round(c*100/len(predictions),3)

def fit_tree(train_labels, train_features, val_labels, val_features, test_labels, test_features):
    n_estimators = np.arange(50, 500, 100)
    max_features = np.arange(0.1, 1.0, 0.2)
    sample_splits = np.arange(2, 11, 2)

    results = []
    for a in n_estimators:
        for b in max_features:
            for c in sample_splits:
                temp = {"n_estimators": a, "max_features": b, "sample_split": c, "train_acc": 0, "val_acc": 0,
                        "test_acc": 0}
                print(f"Computing for {a} {b} {c}")
                classifier = RandomForestClassifier(n_estimators=a, min_samples_split=c, max_features=b, bootstrap=True,
                                             oob_score=True)
                classifier.fit(train_features, train_labels)
                temp["oob_score"] = round(classifier.oob_score_ * 100, 4)
                temp["train_acc"] = get_accuracy(classifier, train_features, train_labels)
                temp["test_acc"] = get_accuracy(classifier, test_features, test_labels)
                temp["val_acc"] = get_accuracy(classifier, val_features, val_labels)
                results.append(temp)

    results_df = pd.DataFrame(results)
    results_df.to_csv('a_b_c_all_combination_grid_search.csv', index=False)

def transform_data(data):
    tranformed_data_dict = []
    cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    median_values = {}
    for col in cols:
        median_values.update({col: train_data[col].describe()["50%"]})

    for row in data.to_dict("records"):
        temp = copy.deepcopy(row)
        for col in cols:
            if (temp[col] > median_values[col]):
                temp[col] = "B"
            else:
                temp[col] = "A"
        tranformed_data_dict.append(temp)

    data_df = pd.DataFrame(tranformed_data_dict)
    return data_df

def get_model_input(train_data, val_data, test_data):

    transformed_train_data = transform_data(train_data)
    transformed_train_data["type"] = "train"
    transformed_val_data = transform_data(val_data)
    transformed_val_data["type"] = "val"
    transformed_test_data = transform_data(test_data)
    transformed_test_data["type"] = "test"

    all_data = pd.DataFrame()
    all_data = all_data.append(transformed_train_data)
    all_data = all_data.append(transformed_val_data)
    all_data = all_data.append(transformed_test_data)

    all_features = pd.get_dummies(all_data)
    train_features = all_features[all_features.type_train == 1]
    test_features = all_features[all_features.type_test == 1]
    val_features = all_features[all_features.type_val == 1]

    train_labels = np.array(transformed_train_data['y'])
    train_features = train_features.drop(["y_no", "y_yes", 'type_test', 'type_train', 'type_val'], axis=1)
    # train_features_list = list(train_features.columns)
    train_features = np.array(train_features)

    val_labels = np.array(transformed_val_data['y'])
    val_features = val_features.drop(["y_no", "y_yes", 'type_test', 'type_train', 'type_val'], axis=1)
    # val_features_list = list(val_features.columns)
    val_features = np.array(val_features)

    test_labels = np.array(transformed_test_data['y'])
    test_features = test_features.drop(["y_no", "y_yes", 'type_test', 'type_train', 'type_val'], axis=1)
    # test_features_list = list(test_features.columns)
    test_features = np.array(test_features)

    return train_labels, train_features, val_labels, val_features, test_labels, test_features


if __name__ == "__main__":

    train_labels, train_features, val_labels, val_features, test_labels, test_features = get_model_input(train_data, val_data, test_data)
    fit_tree(train_labels, train_features, val_labels, val_features, test_labels, test_features)


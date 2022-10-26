from sklearn.neural_network import MLPClassifier
import time
import pandas as pd
import numpy as np

import sys
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]

def get_model_input_data(data):
    labels = data.label.values.tolist()
    y_data = pd.get_dummies(labels).values.tolist()
    data.drop(["label"], axis=1, inplace=True)

    x_data_model = [d for d in data.values.tolist()]
    y_data_model = [d for d in y_data]

    return np.array(x_data_model), np.array(y_data_model), labels

def get_accuracy(y_act, y_pred):
        c = 0
        for i in range(0, len(y_pred)):
            y = np.argmax(y_pred[i])
            if y == y_act[i]:
                c += 1
        return round(100 * c / len(y_pred), 3)


def get_confusion_matrix(y_act, y_pred):
    conf_mat = np.zeros((10, 10))
    for i in range(0, len(y_pred)):
        y = np.argmax(y_pred[i])
        conf_mat[y_act[i]][y] += 1
    return conf_mat

if __name__ == "__main__":

    train_data = pd.read_csv(train_data_path, sep=",", header=None)
    test_data = pd.read_csv(test_data_path, sep=",", header=None)

    train_data.columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
                          'feature_7', 'feature_8', 'feature_9', 'label']

    test_data.columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
                         'feature_7', 'feature_8', 'feature_9', 'label']

    train_data = pd.get_dummies(train_data,
                                columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
                                         'feature_6', 'feature_7', 'feature_8', 'feature_9'])
    test_data = pd.get_dummies(test_data,
                               columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
                                        'feature_6', 'feature_7', 'feature_8', 'feature_9'])

    training_x_data_model, training_y_data_model, training_labels = get_model_input_data(train_data)
    test_x_data_model, test_y_data_model, test_labels = get_model_input_data(test_data)

    classifier = MLPClassifier(hidden_layer_sizes=(85, 100, 100, 10), max_iter=500, activation='relu', solver='sgd',
                               random_state=1, learning_rate_init=0.1, learning_rate="constant", tol=0.015)

    start_time = time.time()
    classifier.fit(training_x_data_model, training_y_data_model)
    print(f"Total train time : {time.time()-start_time}")

    y_pred = classifier.predict(training_x_data_model)
    print(f"Training accuracy : {get_accuracy(training_labels, y_pred)}")

    y_pred = classifier.predict(test_x_data_model)
    print(f"Training accuracy : {get_accuracy(test_labels, y_pred)}")
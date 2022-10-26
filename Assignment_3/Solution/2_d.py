import time
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import copy

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

def draw_plot_acc_vs_hidden_units(hidden_units, train_acc, test_acc):
    x = hidden_units[:]
    plt.figure(figsize=(13, 8))
    plt.plot(x, train_acc, 'r-o', label="training accuracy")
    plt.plot(x, test_acc, 'g-o', label="test accuracy")
    plt.legend()
    plt.xticks(np.arange(min(x), max(x) + 1, 5.0))
    plt.title("Accuracy vs Hidden Units")
    plt.xlabel("# Hidden Units")
    plt.ylabel("Accuracy (%)")
    plt.savefig("acc_vs_hidden_units_d.png")

def draw_plot_time_vs_hidden_units(hidden_units, ttime):
    x = hidden_units[:]
    plt.figure(figsize=(13, 8))
    plt.plot(x, ttime, 'b-o', label="training time")
    plt.legend()
    plt.xticks(np.arange(min(x), max(x) + 1, 5.0))
    plt.title("Training time vs Hidden Units")
    plt.xlabel("# Hidden Units")
    plt.ylabel("Training time (secs)")
    plt.savefig("training_time_vs_acc_d.png")

class DenseLayer():
    def __init__(self, input_dim, output_dim):
        self.weight_matrix = np.random.rand(input_dim, output_dim)

    def forward(self, inp):
        self.input = np.array(inp)
        self.output = self.input @ self.weight_matrix  # + self.bias
        return self.output

    def backward(self, error, lr):
        net_error = error @ self.weight_matrix.T
        weight_error = self.input.T @ error
        self.weight_matrix -= lr * weight_error
        return net_error


class MSELoss():
    def __init__(self):
        pass

    def get_loss(self, actual, predicted):
        return np.mean(np.power(actual - predicted, 2))

    def get_derivative_loss(self, actual, predicted):
        return (np.array(predicted) - actual)/ len(actual)


class ReLuActivation():
    def __init__(self):
        pass

    def get_activation(self, inp):
        ip = copy.deepcopy(inp)
        ip[np.where(ip <= 0)] = 0
        return ip

    def get_activation_derivative(self, inp):
        ip = copy.deepcopy(inp)
        ip[np.where(ip > 0)] = 1
        ip[np.where(ip < 0)] = 0
        return ip


class TanhActivation():
    def __init__(self):
        pass

    def get_activation(self, inp):
        return np.tanh(inp)

    def get_activation_derivative(self, inp):
        return 1 - np.tanh(inp) ** 2


class SigmoidActivation():
    def __init__(self):
        pass

    def get_activation(self, inp):
        return 1 / (1 + np.exp(-inp))

    def get_activation_derivative(self, inp):
        Oj = self.get_activation(inp)
        return Oj * (1 - Oj)


class ActivationLayer():
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def forward(self, inp):
        self.input = inp
        self.output = self.activation_function.get_activation(self.input)
        return self.output

    def backward(self, error, learning_rate):
        return self.activation_function.get_activation_derivative(self.input) * error


class NeuralNetwork:
    def __init__(self, loss_function=MSELoss()):
        self.layers = []
        self.loss_function = loss_function

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x_test):
        y_pred = []
        for data in x_test:
            op = data
            for layer in self.layers:
                op = layer.forward(op)
            y_pred.append(op)

        return y_pred

    def fit(self, x_train, y_train, labels, epochs=1000, learning_rate=0.1, batch_size=2):
        total_batches = int(len(x_train) / batch_size)
        epoch_loss = []

        for i in range(epochs):
            lr = learning_rate/(pow(i+1,0.5))
            #             print(f"Checking for {i}")
            batch_wise_error = 0
            batch_wise_acc = 0
            for batch_number in range(total_batches):
                x_batch_data = x_train[batch_number * batch_size:(batch_number + 1) * batch_size]
                y_batch_data = y_train[batch_number * batch_size:(batch_number + 1) * batch_size]
                batch_labels = labels[batch_number * batch_size:(batch_number + 1) * batch_size]
                batch_pred = x_batch_data[:]
                for layer in self.layers:
                    batch_pred = layer.forward(batch_pred)

                batch_wise_error_for_back_prop = self.loss_function.get_derivative_loss(y_batch_data, batch_pred)
                batch_wise_error += self.loss_function.get_loss(y_batch_data, batch_pred)
                batch_wise_acc += get_accuracy(batch_labels, batch_pred)
                #                 print(f"back prop error ", batch_wise_error_for_back_prop)
                for layer in reversed(self.layers):
                    batch_wise_error_for_back_prop = layer.backward(batch_wise_error_for_back_prop, lr)

            epoch_loss.append(batch_wise_error / total_batches)
#             print(f"Average Loss for epoch : {i + 1} is {batch_wise_error / total_batches}")
#             print(f"Average accuracy for epoch : {i + 1} is {batch_wise_acc / total_batches}")
            if (len(epoch_loss) > 5 and epoch_loss[-1] <= 0.015 and abs(
                    epoch_loss[-1] - np.mean(epoch_loss[-6:-1])) <= 1e-5):
                break


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

    hidden_units = [5, 10, 15, 20, 25]
    train_acc = []
    test_acc = []
    conf_mats = []
    ttime = []
    for hu in hidden_units:
        print(f"Training for hu {hu}")
        model = NeuralNetwork()
        model.add(DenseLayer(85, hu))
        model.add(ActivationLayer(SigmoidActivation()))
        model.add(DenseLayer(hu, 10))
        model.add(ActivationLayer(SigmoidActivation()))

        start_time = time.time()
        model.fit(training_x_data_model, training_y_data_model, training_labels, epochs=10000, learning_rate=0.1,
                  batch_size=100)
        print(f"Total Training time : {time.time() - start_time}")
        ttime.append(time.time() - start_time)

        # Get Training Accuracy
        preds = model.predict(training_x_data_model)
        tacc = get_accuracy(training_labels, preds)
        print(f"Training accuracy : {tacc}")
        train_acc.append(tacc)
#         print(f"Confusion matrix")
#         print(get_confusion_matrix(training_labels, preds))

        preds = model.predict(test_x_data_model)
        tacc = get_accuracy(test_labels, preds)
        print(f"Test accuracy : {tacc}")
        test_acc.append(tacc)
        print(f"Confusion matrix")
        cm = get_confusion_matrix(test_labels, preds)
        print(cm)
        conf_mats.append(cm)
        print("----------------------------------------------------------------------------------------------")

    draw_plot_acc_vs_hidden_units(hidden_units, train_acc, test_acc)
    draw_plot_time_vs_hidden_units(hidden_units, ttime)
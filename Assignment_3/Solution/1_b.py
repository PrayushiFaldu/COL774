import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

import sys
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
val_data_path = sys.argv[3]

train_data = pd.read_csv(train_data_path, sep=";")
val_data = pd.read_csv(val_data_path, sep=";")
test_data = pd.read_csv(test_data_path, sep=";")


class Tree:
    def __init__(self):
        self.data = None
        self.node_val = None
        self.split_attribute_name = None
        self.parent = None
        self.children = {}  # should be dict of form {"val1" : Node, "val2" : Node}
        self.splitted_nodes = []  # should be list of nodes split till now.
        self.splitted_values = []
        self.is_leaf_node = True
        self.total_yes = 0
        self.total_no = 0


def compute_entropy(data):
    p_y = len(data[data["y"] == "yes"]) / len(data)
    if p_y == 0 or p_y == 1:
        return 0
    entropy = -(p_y * math.log2(p_y) + (1 - p_y) * math.log2(1 - p_y))
    return entropy


def is_leaf_node(data, splitted_columns):
    if len(data) == 0:
        return True, 0
    labels = list(data["y"].unique())
    if len(labels) == 1:
        return True, 1
    if len(splitted_columns) == len(list(data.columns)) - 1:
        return True, 2
    return False, 0


def find_ideal_split(data, splitted_nodes):
    H_y = compute_entropy(data)
    splitted_nodes.append('y')
    max_MI = -1
    split_col = None

    for col in list(data.columns):
        if not col in splitted_nodes:
            H_y_x = 0
            unique_values = list(data[col].unique())
            for value in unique_values:
                temp_data = data[data[col] == value]
                entropy = compute_entropy(temp_data)
                H_y_x = H_y_x + entropy * (len(temp_data) / len(data))
            MI = H_y - H_y_x
            if (MI > max_MI):
                max_MI = MI
                split_col = col
    return split_col


def compute_accuracy(tree_root, data):
    predicted = []
    actual = []
    for row in data.to_dict("records"):
        nextnode = tree_root
        not_found_flag = 0
        while (nextnode.is_leaf_node != True):
            if (nextnode.split_attribute_name is None):
                break
            elif (not row[nextnode.split_attribute_name] in nextnode.children):
                not_found_flag = 1
                break
            nextnode = nextnode.children[row[nextnode.split_attribute_name]]

        if (not_found_flag == 1):
            predicted.append("no")
            actual.append(row["y"])
            continue

        else:
            if nextnode.total_yes > nextnode.total_no:
                pred = "yes"
            else:
                pred = "no"
            predicted.append(pred)
            actual.append(row["y"])
            continue

    c = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            c += 1
    return round(c * 100 / len(actual), 3)


def fit_tree(transformed_data):
    all_leaf_nodes = set()
    queue = []
    queue_front = 0

    root = Tree()
    root.data = transformed_data
    root.node_val = "root"
    root.parent = None
    t = root.data["y"].value_counts()
    root.total_yes = t.get("yes", 0)
    root.total_no = t.get("no", 0)
    queue.append(root)

    while (queue_front < len(queue)):
        #     print("############################")
        top_node = queue[:][queue_front]
        queue_front += 1
        if(queue_front%1000 == 0):
            print("Progress :", queue_front)
        bool_leaf, leaf_type = is_leaf_node(top_node.data, top_node.splitted_nodes[:])
        if (bool_leaf):
            if (leaf_type == 0):
                top_node.is_leaf_node = True
                top_node.children.update({"yes": 0, "no": 1})
                top_node.total_yes = 0
                top_node.total_no = 1
                all_leaf_nodes.add(top_node)

            elif (leaf_type == 1):
                top_node.is_leaf_node = True
                all_leaf_nodes.add(top_node)
                t = top_node.data["y"].value_counts()
                top_node.children.update({"yes": t.get("yes", 0), \
                                          "no": t.get("no", 0)})
                top_node.total_yes = t.get("yes", 0)
                top_node.total_no = t.get("no", 0)

            else:
                column = list(top_node.splitted_nodes)[-1]
                for value in list(top_node.data[column].unique()):
                    new_node = Tree()
                    new_node.node_val = value
                    new_node.is_leaf_node = True
                    all_leaf_nodes.add(new_node)
                    t = top_node.data[(top_node.data[column] == value)]["y"].value_counts()
                    new_node.children.update({"yes": t.get("yes", 0), \
                                              "no": t.get("no", 0)})
                    new_node.total_yes = t.get("yes", 0)
                    new_node.total_no = t.get("no", 0)
                    new_node.parent = top_node
                    top_node.children.update({value: new_node})
        else:
            split_attribute = find_ideal_split(top_node.data, top_node.splitted_nodes[:])
            top_node.split_attribute_name = split_attribute
            top_node.is_leaf_node = False
            if top_node in all_leaf_nodes:
                all_leaf_nodes.remove(top_node)
            if (not split_attribute is None):
                for value in list(top_node.data[split_attribute].unique()):
                    new_node = Tree()
                    new_node.data = top_node.data[top_node.data[split_attribute] == value]
                    new_node.splitted_nodes = top_node.splitted_nodes + [split_attribute]
                    new_node.splitted_values = top_node.splitted_values + [value]
                    new_node.node_val = value
                    new_node.parent = top_node
                    t = new_node.data["y"].value_counts()
                    new_node.total_yes = t.get("yes", 0)
                    new_node.total_no = t.get("no", 0)
                    top_node.children.update({value: new_node})
                    queue.append(new_node)

    return root, all_leaf_nodes


def calculate_nodes(troot):
    nodes_list = []
    c = 0
    total_nodes = 0
    nodes_list.append(troot)
    while (c < len(nodes_list)):
        top_node = nodes_list[c]
        c += 1
        total_nodes += 1
        if (top_node.is_leaf_node == False):
            for child in top_node.children.values():
                if not child is None:
                    nodes_list.append(child)
    return total_nodes


def post_prune_tree(root, all_leaf_nodes):
    all_leaf_nodes_pruning = list(all_leaf_nodes)[:]
    max_accuracy = compute_accuracy(root, transformed_val_data)
    all_leaf_nodes_pruning = list(all_leaf_nodes_pruning)
    parent_nodes = [node.parent for node in all_leaf_nodes_pruning]

    removed_lf = set()
    c = 0
    new_leaf_nodes = 0
    initial_lf = len(parent_nodes)

    train_acc = []
    test_acc = []
    val_acc = []
    node_count = []
    while (c < len(parent_nodes)):
        # print("-----------------------------")
        if (c % 1000 == 0):
            print("Progress :", c)
        # print(f"Total nodes : {len(parent_nodes)}")
        lf_node = parent_nodes[c]
        if (not lf_node in removed_lf) and (not lf_node is None):
            curr_parent_node = lf_node.parent
            if (not curr_parent_node in parent_nodes[c:]) and (not curr_parent_node is None):
                parent_nodes.append(curr_parent_node)
                new_leaf_nodes += 1
                curr_parent_node.is_leaf_node = True
                temp_accuracy = compute_accuracy(root, transformed_val_data)
                if (temp_accuracy > max_accuracy):
                    max_accuracy = temp_accuracy
                    for child in curr_parent_node.children.values():
                        if not child is None:
                            removed_lf.add(child)

                    print(f"Max accuracy : {max_accuracy}")
                    print(f"Total leaf nodes = {initial_lf - len(removed_lf) + new_leaf_nodes}")
                    train_acc.append(compute_accuracy(root, transformed_train_data))
                    val_acc.append(max_accuracy)
                    test_acc.append(compute_accuracy(root, transformed_test_data))
                    node_count.append(calculate_nodes(root))

                else:
                    for child in curr_parent_node.children.values():
                        if not child is None:
                            removed_lf.add(child)
                    lf_node.parent.is_leaf_node = False
        c += 1

    return train_acc, val_acc, test_acc, node_count


def transform_data(data, is_one_hot_encoded=False):
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

    if (is_one_hot_encoded):
        cols = data_df.columns.tolist()
        cols.remove("y")
        data_df = pd.get_dummies(data_df, columns=cols)

    return data_df


def draw_plot(train_acc, val_acc, test_acc, node_count):
    x = node_count[:]
    plt.figure(figsize=(13, 8))
    plt.plot(x, train_acc, 'r-', label="training")
    plt.plot(x, val_acc, 'y-', label="validation")
    plt.plot(x, test_acc, 'g-', label="test")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.title("Pruned Tree Accuracy vs number of nodes.")
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    plt.savefig("1_b_post_prune.png")


if __name__ == "__main__":
    is_one_hot_encoded = False
    transformed_train_data = transform_data(train_data, is_one_hot_encoded)
    transformed_val_data = transform_data(val_data, is_one_hot_encoded)
    transformed_test_data = transform_data(test_data, is_one_hot_encoded)
    print("Fitting Tree")
    root, all_leaf_nodes = fit_tree(transformed_train_data)
    print("Pruning Tree")
    train_acc, val_acc, test_acc, node_count = post_prune_tree(root, all_leaf_nodes)
    draw_plot(train_acc, val_acc, test_acc, node_count)

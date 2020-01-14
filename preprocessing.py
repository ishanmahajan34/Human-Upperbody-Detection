import numpy as np
import pandas as pd
import pickle

def dict_to_matrix(data_dict, labels):
    matrix = []
    for label in labels:
        matrix.append(data_dict[label])
    
    return np.array(matrix).T


def get_labels(data):
    labels = []
    for index, value in enumerate(data):
        labels.append(value[0])    
    return labels


X_train_data = pickle.load(open('cse512hw4/Train_Features.pkl','rb'), encoding='latin')
X_val_data = pickle.load(open('cse512hw4/Val_Features.pkl','rb'), encoding='latin')

# Data in csv format
y_train_data = pd.read_csv('cse512hw4/Train_Labels.csv').to_numpy()
y_val_data = pd.read_csv('cse512hw4/Val_Labels.csv').to_numpy()


y_train_labels = get_labels(y_train_data)
X_train = dict_to_matrix(X_train_data, y_train_labels)
y_train = y_train_data[:, 1]

y_val_labels = get_labels(y_val_data)
X_val = dict_to_matrix(X_val_data, y_val_labels)
y_val = y_val_data[:, 1]

x_train_df = pd.DataFrame(X_train) 
x_val_df = pd.DataFrame(X_val) 
y_train_df = pd.DataFrame(y_train)
y_val_df = pd.DataFrame(y_val) 

x_train_df.to_csv(r'x_train.csv', index=None, header=False)
x_val_df.to_csv(r'x_val.csv', index=None, header=False)
y_train_df.to_csv(r'y_train.csv', index=None, header=False)
y_val_df.to_csv(r'y_val.csv', index=None, header=False)
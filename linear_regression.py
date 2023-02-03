print("working")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print("all imports are working")

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training data
#we use the above data to predict how is going to survive and who is not,
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')   #testing data
print(dftrain.head())
print("data printed")

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
print("data printed but we can see that survived got poped")

print("/n", y_train.head())

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town','alone']
NUMERIC_COLUMN = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
print(feature_columns)
print("\nthe above data is from feature_columns with only with vocabulary data only\n\n")

for feature_name in NUMERIC_COLUMN:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)
print("\nthe above data is from feature_columns with only with numerical data only\n\n")

# creating a model

#before that the training process....



#input function (read text file (imp))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=12):
    def input_function(): # inner function, will be returened 
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000) # randomize order of data
        

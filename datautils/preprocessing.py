import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# https://archive.ics.uci.edu/ml/datasets/mushroom
# It contains 22 features and 1 output variable.
datasets_dir = os.path.abspath(os.path.dirname(__file__))
mushroom_data_path = os.path.join(datasets_dir, "mushroom/mushroom.csv")


def load_data(name="mushroom"):
    if name == "mushroom":
        # do something
        df = pd.read_csv(mushroom_data_path)
        X, y = preprocess_data(df)
    else:
        raise Exception("Undefined dataset {}".format(name))

    return X, y


def preprocess_data(df):
    # The first column of the mushrooms dataset are the labels.
    # e = edible, p = poisonous

    # Get the data with only the features (i.e. without labels).
    # df.iloc[:, 1:].shape == (8123, 22), where 22 is the number of input features and 8123 is the number of
    # observations.

    # df_.shape = (8123, 117)
    df_ = pd.get_dummies(df.iloc[:, 1:]) # I think this converts the input features to a categorical representation
    _, X = df_.columns, df_.values

    # Get only the labels.
    y = df.iloc[:, 0].values
    label_encoder_y = LabelEncoder()  # Convert the labels to numbers.
    y = label_encoder_y.fit_transform(y)

    return X, y

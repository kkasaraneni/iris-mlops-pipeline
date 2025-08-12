# src/preprocess.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = df.columns.str.replace(' (cm)', '', regex=False)  # remove cm unit

    # Split
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

if __name__ == "__main__":
    load_and_preprocess()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import codes.config as config


if __name__ == '__main__':

    working_dataset = config.PERSONALITY_CSV
    X, y = None, None

    # Reading the dataset
    df = pd.read_csv(working_dataset, encoding="latin")

    # Dropping irrelevant columns
    if working_dataset == config.ESSAY_CSV:
        X = df.drop(columns=['cNEU', '#AUTHID', 'TEXT'])

    elif working_dataset == config.PERSONALITY_CSV:
        X = df.drop(columns=['cNEU', 'DATE', '#AUTHID', 'STATUS'])

    y = df['cNEU']

    # Performing one-hot encoding on binary flags
    X = pd.get_dummies(X, columns=['cEXT', 'cAGR', 'cCON', 'cOPN'], drop_first=True)

    X.dropna(inplace=True)
    y = y[X.index]  # Match the indices of X and y after removing rows with missing values

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    print(X_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    #
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    #
    # Visualizing Feature Importance

    # Get feature importances
    feature_importances = clf.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), feature_importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.show()

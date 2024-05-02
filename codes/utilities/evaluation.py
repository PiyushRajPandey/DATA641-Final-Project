import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score


def k_fold_cross_val_evaluation(classifier, param_grid, kfold, X_train, y_train, verbatim):
    if verbatim:
        print("Evaluating using cross-validation and hyperparameter tuning")

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    classifier = grid_search.best_estimator_

    accuracy_scores = cross_val_score(
        classifier,
        X_train, y_train,
        cv=kfold, scoring='accuracy'
    )

    acc_means = round(np.mean(accuracy_scores), 5)
    acc_std_devs = round(np.std(accuracy_scores), 5)

    if verbatim:
        print("\n\tmean accuracy = {}, "
              "\n\tstdev accuracy = {}\n\n"
              .format(acc_means, acc_std_devs))


def test_set_evaluation(training_vectorizer, classifier, X_train, y_train, X_test, y_test):
    # Word Embeddings
    if training_vectorizer is None:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    # TF-IDF and Count Vectorizer
    else:
        X_features_test = training_vectorizer.transform(X_test)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_features_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 3))
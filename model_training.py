# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_models(X, y):
    # Splitting data for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))

    # Train K Nearest Neighbors classifier
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    knn_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))

    # Train Support Vector Machine classifier
    svm_classifier = SVC(kernel='rbf', random_state=42)
    svm_classifier.fit(X_train, y_train)
    svm_accuracy = accuracy_score(y_test, svm_classifier.predict(X_test))

    # Train Logistic Regression classifier
    logreg_classifier = LogisticRegression(random_state=42)
    logreg_classifier.fit(X_train, y_train)
    logreg_accuracy = accuracy_score(y_test, logreg_classifier.predict(X_test))

    # Determine the classifier with the highest accuracy
    classifiers = {
        'Random Forest': (rf_classifier, rf_accuracy),
        'K Nearest Neighbors': (knn_classifier, knn_accuracy),
        'Support Vector Machine': (svm_classifier, svm_accuracy),
        'Logistic Regression': (logreg_classifier, logreg_accuracy)
    }
    best_classifier_name = max(classifiers, key=lambda k: classifiers[k][1])
    best_classifier = classifiers[best_classifier_name][0]

    return best_classifier, classifiers

import os
from io import StringIO

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
except Exception as e:
    print("Missing required packages:", e)
    print("Install with: pip install pandas scikit-learn")
    raise


def load_iris_from_py(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # Join and treat as CSV
    s = "\n".join(lines)
    df = pd.read_csv(StringIO(s), header=None,
                     names=['sepal_len','sepal_wid','petal_len','petal_wid','label'])
    return df


def main():
    base = os.path.dirname(__file__)
    iris_py = os.path.join(base, 'iris.py')
    if not os.path.exists(iris_py):
        print(f"Data file not found at {iris_py}")
        return

    df = load_iris_from_py(iris_py)
    X = df[['sepal_len','sepal_wid','petal_len','petal_wid']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, preds))


if __name__ == '__main__':
    main()

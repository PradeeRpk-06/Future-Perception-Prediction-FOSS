import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from visualization import plot_actual_vs_predicted  
from summary import analyze_classification_results  

def run_classification(csv_path="clean_reviews.csv"):
    import pandas as pd
    import numpy as np
    import ast
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    df = pd.read_csv(csv_path)

    df["fnet_embedding"] = df["fnet_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    X = np.vstack(df["fnet_embedding"].values)
    y = df["overall"]

    y_num = y.apply(lambda r: -1 if r <= 2 else (0 if r == 3 else 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.50, random_state=42)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": float(acc),
        "summary": {
            "total_samples": len(y_test),
            "correct_pre_predictions": int((y_pred == y_test).sum()),
            "wrong_predictions": int((y_pred != y_test).sum()),
        },
        "report": report
    }

from pathlib import Path
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib

from cover_type_prediction.config import MODELS_DIR

model_version = Path(__file__).parent.name

def main():
    # Create model output directory if needed
    model_output_dir = MODELS_DIR / model_version
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    covertype_data = fetch_covtype(as_frame=True)

    covertype_df = covertype_data.data
    covertype_df['target'] = covertype_data.target

    X = covertype_df.drop('target', axis=1)
    y = covertype_df['target']

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # train model
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=True
    )
    gbm.fit(X_train, y_train)

    # evaluate model
    y_pred = gbm.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision (macro): {precision:.2f}")
    print(f"Recall (macro): {recall:.2f}")
    print(f"F1 Score (macro): {f1:.2f}")

    # Save model and training results
    with open(model_output_dir / 'training_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    joblib.dump(gbm, model_output_dir / 'covertype_gbm_model.joblib')




if __name__ == "__main__":
    main()




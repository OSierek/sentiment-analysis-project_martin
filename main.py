from src.save_model import save_model
from src.train import load_and_validate_data, split_data, train_model


def main(data_path: str, model_path: str) -> None:
    """
    Main workflow to load, train, evaluate, and save the model.
    """
    df = load_and_validate_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    clf = train_model(X_train, y_train)

    # Evaluate and print accuracy
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    save_model(clf, model_path)
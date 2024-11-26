import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate synthetic data
def generate_data(n_samples=1000, n_features=10):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train and log models using MLflow
def train_and_log_model():
    
    tracking_uri = "file:/app/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # Set or create experiment
    experiment_name = "Classification_Tuning_Experiment"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"Using experiment '{experiment.name}' with ID {experiment.experiment_id}")

    # Split data
    X_train, X_test, y_train, y_test = generate_data()

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }

    # Create base model
    model = RandomForestClassifier(random_state=42)

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    # Log model and parameters to MLflow
    with mlflow.start_run() as run:
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(best_model, "model")
        print("Model logged with parameters:", grid_search.best_params_)

        # Register model in MLflow Model Registry
        model_name = "model"
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            print(f"Model '{model_name}' already exists in the registry.")

        # Create a new version of the model
        model_uri = f"runs:/{run.info.run_id}/model"
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run.info.run_id
        )

        # Transition the model version to "Production"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        print(f"Model version {model_version.version} is now in 'Production' stage.")

    # Output results
    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    print(classification_report(y_test, best_model.predict(X_test)))

if __name__ == "__main__":
    train_and_log_model()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os
from src.utils import load_diabetes_data, preprocess_diabetes_data, split_features_target, split_train_test

def perform_eda(df):
    """
    Performs Exploratory Data Analysis (EDA) on the preprocessed DataFrame.
    (Simplified for app.py; full EDA details are in explore.ipynb)
    """
    print("\n--- Starting EDA Summary ---")
    print("DataFrame Info:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nNull values after preprocessing:")
    print(df.isnull().sum())

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Outcome', data=df)
    plt.title('Distribution of Outcome Variable')
    plt.savefig('data/processed/outcome_distribution.png')
    plt.close()
    print("EDA plots saved to data/processed/")
    print("--- EDA Summary Complete ---")

def train_and_evaluate_tree(X_train, y_train, X_test, y_test, criterion='gini', model_name="Decision Tree"):
    """
    Trains a Decision Tree Classifier and evaluates its performance.
    """
    print(f"\n--- Training and Evaluating {model_name} with {criterion.capitalize()} Impurity ---")
    model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    return model, accuracy, report, conf_matrix

def optimize_decision_tree(X_train, y_train, criterion='entropy'):
    """
    Optimizes a Decision Tree Classifier using GridSearchCV.
    """
    print(f"\n--- Optimizing Decision Tree with {criterion.capitalize()} Impurity using GridSearchCV ---")
    dt_base = DecisionTreeClassifier(criterion=criterion, random_state=42)

    param_grid = {
        'max_depth': [3, 4, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
    }

    grid_search = GridSearchCV(estimator=dt_base, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def save_model(model, filename="best_diabetes_decision_tree_model.joblib"):
    """
    Saves the trained model to the 'models' directory.
    """
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, filename)
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to: {model_path}")

def plot_optimized_tree(model, X_columns, class_names, filename="optimized_decision_tree.png"):
    """
    Plots and saves the optimized decision tree.
    """
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    plt.figure(figsize=(25, 15))
    plot_tree(model, feature_names=X_columns, class_names=class_names, filled=True, rounded=True, fontsize=10)
    plt.title("Optimized Decision Tree")
    plt.savefig(os.path.join(model_dir, filename))
    plt.close()
    print(f"Optimized tree plot saved to: {os.path.join(model_dir, filename)}")

# --- Main execution block ---
if __name__ == "__main__":
    # Step 1: Load the dataset
    df = load_diabetes_data()
    if df is None:
        exit("Failed to load data. Exiting.")

    # Step 2: Perform Data Processing (EDA summary in app, full EDA in explore.ipynb)
    processed_df = preprocess_diabetes_data(df)
    if processed_df is None:
        exit("Failed to preprocess data. Exiting.")

    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_csv('data/processed/diabetes_processed.csv', index=False)
    print("Processed data saved to data/processed/diabetes_processed.csv")

    perform_eda(processed_df)

    X, y = split_features_target(processed_df, target_column='Outcome')
    if X is None:
        exit("Failed to split features and target. Exiting.")

    X_train, X_test, y_train, y_test = split_train_test(X, y, stratify=y)
    if X_train is None:
        exit("Failed to split train and test data. Exiting.")

    # Step 3: Build and analyze decision trees (unoptimized)
    _, _, _, _ = train_and_evaluate_tree(X_train, y_train, X_test, y_test, criterion='gini', model_name="Decision Tree (Unoptimized Gini)")
    _, _, _, _ = train_and_evaluate_tree(X_train, y_train, X_test, y_test, criterion='entropy', model_name="Decision Tree (Unoptimized Entropy)")

    # Step 4: Optimize the model (using Entropy as it showed slightly better recall for Class 1)
    best_dt_model = optimize_decision_tree(X_train, y_train, criterion='entropy')

    print("\n--- Evaluation of Optimized Decision Tree Model ---")
    y_pred_optimized = best_dt_model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_optimized))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_optimized))

    plot_optimized_tree(best_dt_model, X.columns.tolist(), ['No Diabetes', 'Diabetes'])

    # Step 5: Save the model
    save_model(best_dt_model)

    print("\nProject execution complete. Check 'data/processed/' and 'models/' folders for outputs.")
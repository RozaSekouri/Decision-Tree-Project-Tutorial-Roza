from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine
import pandas as pd
from sklearn.model_selection import train_test_split

def load_diabetes_data(url="https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"):
    """
    Loads the diabetes dataset from a given URL.
    """
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_diabetes_data(df):
    """
    Performs initial data preprocessing for the diabetes dataset:
    - Replaces '0' values in specific columns with NaN.
    - Imputes NaN values with the median of their respective columns.
    """
    if df is None:
        return None

    # Columns where 0 typically represents a missing value in this dataset
    cols_with_zeros_as_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # Replace 0s with NaN
    for col in cols_with_zeros_as_nan:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(0, pd.NA)

    # Impute NaNs with the median
    for col in cols_with_zeros_as_nan:
        if col in df_processed.columns and df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)

    print("Data preprocessing complete.")
    return df_processed

def split_features_target(df, target_column='Outcome'):
    """
    Splits the DataFrame into features (X) and target (y).
    """
    if df is None:
        return None, None
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("Features and target split.")
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Splits the features and target into training and testing sets.
    """
    if X is None or y is None:
        return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    print(f"Data split into training (shape: {X_train.shape}) and testing (shape: {X_test.shape}) sets.")
    return X_train, X_test, y_train, y_test
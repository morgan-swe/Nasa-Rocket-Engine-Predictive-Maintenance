import pandas as pd
import config
from pathlib import Path

COLUMN_NAMES = [
    'unit_number',
    'time_in_cycles',
    'operational_setting_1',
    'operational_setting_2',
    'operational_setting_3',
    'sensor_measurement_1',
    'sensor_measurement_2',
    'sensor_measurement_3',
    'sensor_measurement_4',
    'sensor_measurement_5',
    'sensor_measurement_6',
    'sensor_measurement_7',
    'sensor_measurement_8',
    'sensor_measurement_9',
    'sensor_measurement_10',
    'sensor_measurement_11',
    'sensor_measurement_12',
    'sensor_measurement_13',
    'sensor_measurement_14',
    'sensor_measurement_15',
    'sensor_measurement_16',
    'sensor_measurement_17',
    'sensor_measurement_18',
    'sensor_measurement_19',
    'sensor_measurement_20',
    'sensor_measurement_21',
]

def load_testing_data() -> pd.DataFrame:
    """
    Load the FD001 testing dataset

    Returns:
    pandas.DataFrame 
        A DataFrame containing the loaded testing data.
    """

    file_path = config.test_fd001

    # Load the FD001 testing dataset from the specified file path.
    try:
        test_df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMN_NAMES)

    except FileNotFoundError:
        raise FileNotFoundError(f"Test file not found: {file_path}. Please check the file path and try again.")

    # Check if the DataFrame is empty and print a message if it is.
    if test_df.empty:
        print(f"The test data file at {file_path} is empty. Please check the file content.")
        return None

    # Verify that the DataFrame has the expected number of columns (26).
    if len(test_df.columns) != 26:
        print(f"Unexpected number of columns in test data. Expected 26, got {len(test_df.columns)}.")
        return None

    # Check for any missing values in the DataFrame and print a message if any are found.
    if test_df.isnull().values.any():
        print(f"Missing values found in test data. Please check the file content.")
        return None

    # Check for duplicate rows in the DataFrame and print a message if any are found. 
    if test_df.duplicated().any():
        print(f"Duplicate rows found in test data. Please check the file content.")
        return None

    # Check for the number of engines (unit numbers) in the test data and print a message if it is unexpected. 
    unique_engines = test_df['unit_number'].nunique()
    if unique_engines != 100:
        print(f"Unexpected number of engines in test data. Expected 100, got {unique_engines}.")
        return None 

    # Return the loaded test DataFrame if all checks pass.
    return test_df


def load_training_data() -> pd.DataFrame:
    """
    Load the FD001 training dataset
    
    Returns:
    pandas.DataFrame 
        A DataFrame containing the loaded training data.
    """

    file_path = config.train_fd001

    # Load the FD001 training dataset from the specified file path. 
    try:
        train_df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMN_NAMES)

    except FileNotFoundError:
        raise FileNotFoundError(f"Train file not found: {file_path}. Please check the file path and try again.")

    # Check if the DataFrame is empty and print a message if it is. 
    if train_df.empty:
        print(f"The train data file at {file_path} is empty. Please check the file content.")
        return None  # Return None if the DataFrame is empty

    # Verify that the DataFrame has the expected number of columns (26). 
    if len(train_df.columns) != 26:
        print(f"Unexpected number of columns in train data. Expected 26, got {len(train_df.columns)}.")
        return None  # Return None if the number of columns is unexpected

    # Check for any missing values in the DataFrame and print a message if any are found. 
    if train_df.isnull().values.any():
        print(f"Missing values found in train data. Please check the file content.")
        return None  # Return None if there are missing values

    # Check for duplicate rows in the DataFrame and print a message if any are found. 
    if train_df.duplicated().any():
        print(f"Duplicate rows found in train data. Please check the file content.")
        return None  # Return None if there are duplicate rows

    # Check for the number of engines (unit numbers) in the training data and print a message if it is unexpected. 
    unique_engines = train_df['unit_number'].nunique()
    if unique_engines != 100:
        print(f"Unexpected number of engines in train data. Expected 100, got {unique_engines}.")
        return None

    # Return the loaded training DataFrame if all checks pass. 
    return train_df

def load_rul_data() -> pd.DataFrame:
    """
    Load the FD001 RUL dataset
    
    Returns:
    pandas.DataFrame 
        A DataFrame containing the loaded RUL data.
    """
  
    file_path = config.rul_fd001

    # Load the FD001 RUL dataset from the specified file path. 
    try:
        rul_df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['RUL'])

    except FileNotFoundError:
        raise FileNotFoundError(f"RUL file not found: {file_path}. Please check the file path and try again.")

    # Check if the DataFrame is empty and print a message if it is. 
    if rul_df.empty:
        print(f"The RUL data file at {file_path} is empty. Please check the file content.")
        return None 

    # Verify that the DataFrame has the expected number of columns (1). 
    if len(rul_df.columns) != 1:
        print(f"Unexpected number of columns in RUL data. Expected 1, got {len(rul_df.columns)}.")
        return None

    # Check for any missing values in the DataFrame and print a message if any are found. 
    if rul_df.isnull().values.any():
        print(f"Missing values found in RUL data. Please check the file content.")
        return None

    # Check for duplicate rows in the DataFrame and print a message if any are found. 
    if rul_df.duplicated().any():
        print(f"Duplicate rows found in RUL data. Please check the file content.")
        return None

    # Return the loaded RUL DataFrame if all checks pass.
    return rul_df

def save_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Save the loaded DataFrames to CSV files in the interim folder.
    """
    output_dir = Path(config.interim_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the test, training, and RUL data using the defined functions.
    test_df = load_testing_data()
    train_df = load_training_data()
    rul_df = load_rul_data()

    # Check if any of the loaded DataFrames are None (indicating a loading error) and print an error message if so. 
    test_df.to_csv(output_dir / 'test_FD001.csv', index=False)
    train_df.to_csv(output_dir / 'train_FD001.csv', index=False)
    rul_df.to_csv(output_dir / 'RUL_FD001.csv', index=False)


def main():
    save_dataframe()
    print("Data saved successfully.")


if __name__ == "__main__":
    main()
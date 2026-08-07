import pandas as pd
from pathlib import Path

column_names = [
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


def load_data():
    # Load the test, train, and RUL data from the specified file paths.
    test_path = 'data/raw/test_FD001.txt'
    train_path = 'data/raw/train_FD001.txt'
    rul_path = 'data/raw/RUL_FD001.txt'

    # Read the data files into pandas DataFrames with appropriate column names.
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=column_names)
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=column_names)
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    # Save the cleaned data to the interim folder for downstream use.
    output_dir = Path('data/interim')
    output_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_dir / 'test_FD001.csv', index=False)
    train_df.to_csv(output_dir / 'train_FD001.csv', index=False)
    rul_df.to_csv(output_dir / 'RUL_FD001.csv', index=False)

    # Return the loaded DataFrames for test, train, and RUL data.
    return test_df, train_df, rul_df


if __name__ == "__main__":
    # Test execution
    train, test, rul = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"RUL shape:   {rul.shape}")
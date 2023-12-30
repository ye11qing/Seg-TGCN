import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os 

class DataProcessor:
    """
    This class is used for processing data from an Excel file.
    """
    def __init__(self, file_path):
        """
        Initialize the DataProcessor with the path to an Excel file.
        """
        self.df = pd.read_csv(file_path)
        self.df = self.df.iloc[:,1:]

    def fill_missing_values_knn(self, time_series, n_neighbors=5):
        """
        Fill missing values in a time series using K-Nearest Neighbors.

        Parameters:
        time_series (numpy.ndarray): The time series data.
        n_neighbors (int): The number of neighbors to use for KNN imputation.

        Returns:
        numpy.ndarray: The time series with missing values filled.
        """
        # Replace 0 m/s with NaN
        time_series[time_series == 0] = np.nan

        # Reshape the time series to fit the KNNImputer
        time_series = time_series.reshape(-1, 1)

        # Fill missing values using KNN
        imputer = KNNImputer(n_neighbors=n_neighbors)
        time_series = imputer.fit_transform(time_series)

        return time_series

    def process_data(self):
        """
        Process the data by filling missing values in each column of the DataFrame.

        Returns:
        pandas.DataFrame: The processed DataFrame.
        """
        for column in self.df.columns:
            self.df[column] = self.fill_missing_values_knn(self.df[column].values)

        return self.df

os.chdir(os.path.dirname(os.path.realpath(__file__)))
processor = DataProcessor("sz_speed.csv")
df = processor.process_data()
df.to_csv("fill_missing_sz_speed.csv")



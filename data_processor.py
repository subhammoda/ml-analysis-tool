import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from utils import detect_problem_type

class DataProcessor:
    """
    A class to handle data loading, preprocessing, and basic analysis.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the DataProcessor with a file path.
        
        Args:
            filepath (str): Path to the data file
        """
        self.filepath = filepath
        self.data = None
        self.original_data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from the file based on its extension."""
        try:
            if self.filepath.endswith('.csv'):
                self.data = pd.read_csv(self.filepath)
            elif self.filepath.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(self.filepath)
            else:
                raise ValueError("Unsupported file format")
            
            self.original_data = self.data.copy()
            logging.info(f"Data loaded successfully: {self.data.shape}")
        
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dict: Information about the dataset including shape, columns, data types, etc.
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_columns": self.data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.data.select_dtypes(include=['object']).columns.tolist(),
            "sample_data": self.data.sample(10).to_dict('records')
        }
        
        # Add basic statistics for numeric columns
        if info["numeric_columns"]:
            info["numeric_stats"] = self.data[info["numeric_columns"]].describe().to_dict()
        
        return info
    
    def preprocess_data(self, target_column: str, drop_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for machine learning.
        
        Args:
            target_column (str): Name of the target column
            drop_columns (Optional[List[str]]): List of columns to drop from features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        logging.info("Starting data preprocessing")
        logging.info(f"  Target column: {target_column}")
        logging.info(f"  Original data shape: {self.data.shape}")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Create a copy for preprocessing
        data = self.data.copy()
        
        # Handle missing values
        logging.info("  Handling missing values")
        data = self._handle_missing_values(data)
        logging.info(f"  Missing values handled - Shape after: {data.shape}")
        
        # Separate features and target before encoding
        X = data.drop(columns=[target_column])
        y = data[target_column]
        # Drop user-specified columns if provided
        if drop_columns:
            drop_actual = [col for col in drop_columns if col in X.columns]
            if drop_actual:
                X = X.drop(columns=drop_actual)
                logging.info(f"  Dropped user-specified columns: {drop_actual}")
        logging.info(f"  Separated features ({X.shape[1]} columns) and target")
        
        # Handle categorical variables in features (one-hot encoding)
        logging.info("  Encoding categorical features")
        original_features_count = X.shape[1]
        X = self._encode_categorical_features(X)
        logging.info(f"  Feature encoding completed - Final features: {X.shape[1]} columns")
        
        # Store original feature count for feature engineering report
        X.attrs['original_features_count'] = original_features_count
        
        # Handle categorical target variable (label encoding)
        if y.dtype == 'object':
            logging.info("  Encoding categorical target variable")
            y = self._encode_target_variable(y)
            logging.info(f"  Target encoding completed - Unique values: {len(y.unique())}")
        
        logging.info(f"  Final feature shape: {X.shape}")
        logging.info(f"  Target variable info: Type: {y.dtype}, Unique values: {len(y.unique())}")
        logging.info(f"Data preprocessing completed")
        
        return X, y
    
    def _get_missing_data_info(self, data: pd.DataFrame) -> Dict:
        """
        Calculate missing data information for a dataframe.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Missing data information
        """
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        columns_with_high_missing = missing_percentage[missing_percentage > 50].index.tolist()
        
        return {
            'missing_percentage': missing_percentage,
            'columns_with_high_missing': columns_with_high_missing,
            'high_missing_count': len(columns_with_high_missing)
        }
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        Drop columns with more than 50% missing data.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        # Get missing data information
        missing_info = self._get_missing_data_info(data)
        columns_to_drop = missing_info['columns_with_high_missing']
        
        # Drop columns with more than 50% missing data
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
            logging.info(f"    Dropped columns with >50% missing data: {columns_to_drop}")
        
        # For remaining numeric columns, fill with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
                logging.info(f"    Filled {col} with median")
        
        # For remaining categorical columns, fill with mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown', inplace=True)
                logging.info(f"    Filled {col} with mode")
        
        return data
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding with one column dropped to prevent multicollinearity.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        from sklearn.preprocessing import OneHotEncoder
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            # Create one-hot encoder with drop='first' to prevent multicollinearity
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            
            # Fit and transform categorical columns
            encoded_features = ohe.fit_transform(X[categorical_columns])
            
            # Create feature names for encoded columns
            feature_names = []
            for i, col in enumerate(categorical_columns):
                categories = ohe.categories_[i][1:]  # Skip first category (dropped)
                feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Create dataframe with encoded features
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
            
            # Drop original categorical columns and concatenate with encoded features
            X_numeric = X.drop(columns=categorical_columns)
            X_encoded = pd.concat([X_numeric, encoded_df], axis=1)
            
            logging.info(f"    Encoded categorical features: {categorical_columns}")

            return X_encoded
        
        return X
    
    def _encode_target_variable(self, y: pd.Series) -> pd.Series:
        """
        Encode categorical target variable using label encoding.
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            pd.Series: Encoded target variable
        """
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        
        return pd.Series(y_encoded, index=y.index)
    
    def get_data_quality_report(self) -> Dict:
        """
        Generate a comprehensive data quality report.
        
        Returns:
            Dict: Data quality report
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        # Get missing data information using helper method
        missing_info = self._get_missing_data_info(self.data)
        
        report = {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "missing_data_percentage": (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
            "duplicate_rows": self.data.duplicated().sum(),
            "numeric_columns_count": len(self.data.select_dtypes(include=[np.number]).columns),
            "categorical_columns_count": len(self.data.select_dtypes(include=['object']).columns),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns_with_high_missing": missing_info['columns_with_high_missing'],
            "high_missing_count": missing_info['high_missing_count']
        }
        
        return report 
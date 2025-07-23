import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
import logging
from utils import detect_problem_type

class FeatureEngineer:
    """
    A class to handle feature engineering operations.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer with default settings."""
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_engineering_steps = []
    
    def engineer_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str = 'auto') -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive feature engineering.
        
        Args:
            X (pd.DataFrame): Feature matrix (already encoded from data processor)
            y (pd.Series): Target variable
            problem_type (str): 'classification', 'regression', or 'auto'
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Engineered features and engineering report
        """
        logging.info("Starting feature engineering process")
        logging.info(f"  Input shape: {X.shape}")
        logging.info(f"  Target variable: {y.name}, Unique values: {len(y.unique())}")
        
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y)
            logging.info(f"  Detected problem type: {problem_type}")
        
        # Track feature counts at different stages
        features_after_encoding = X.shape[1]
        original_features_count = getattr(X, 'attrs', {}).get('original_features_count', features_after_encoding)
        
        engineering_report = {
            'original_features': original_features_count,
            'features_after_encoding': features_after_encoding,
            'problem_type': problem_type,
            'steps_applied': [],
            'feature_importance': {},
            'correlation_analysis': {},
            'outlier_analysis': {}
        }
        
        # Step 1: Handle outliers
        logging.info("  Step 1: Handling outliers")
        X_cleaned = self._handle_outliers(X, engineering_report)
        logging.info(f"  Outlier handling completed")
        
        # Step 2: Feature scaling
        logging.info("  Step 2: Feature scaling")
        X_scaled = self._scale_features(X_cleaned, engineering_report)
        logging.info(f"  Feature scaling completed")
        
        # Step 3: Feature selection
        logging.info("  Step 3: Feature selection")
        X_selected = self._select_features(X_scaled, y, problem_type, engineering_report)
        logging.info(f"  Feature selection completed")
        
        # Step 4: Dimensionality reduction (if needed)
        logging.info("  Step 4: Dimensionality reduction")
        X_final = self._reduce_dimensions(X_selected, engineering_report)
        logging.info(f"  Dimensionality reduction completed")
        
        engineering_report['final_features'] = X_final.shape[1]
        engineering_report['feature_reduction'] = features_after_encoding - X_final.shape[1]
        
        logging.info(f"  Final shape: {X_final.shape}")
        logging.info(f"  Features reduced: {engineering_report['feature_reduction']}")
        logging.info(f"  Steps applied: {', '.join(engineering_report['steps_applied'])}")
        logging.info(f"Feature engineering completed")
        
        return X_final, engineering_report
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """
        Automatically detect if the problem is classification or regression.
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            str: 'classification' or 'regression'
        """
        return detect_problem_type(y)
    
    def _handle_outliers(self, X: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """
        Handle outliers using IQR method for numeric features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            report (Dict): Engineering report to update
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        X_cleaned = X.copy()
        outlier_counts = {}
        
        for column in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((X[column] < lower_bound) | (X[column] > upper_bound)).sum()
            outlier_counts[column] = outliers
            
            # Cap outliers instead of removing them
            X_cleaned[column] = X_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
        
        report['outlier_analysis'] = outlier_counts
        report['steps_applied'].append('outlier_handling')
        
        return X_cleaned
    
    def _scale_features(self, X: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """
        Scale features using appropriate scaling method based on data characteristics.
        
        Args:
            X (pd.DataFrame): Feature matrix
            report (Dict): Engineering report to update
            
        Returns:
            pd.DataFrame: Scaled features
        """
        
        # Get numeric columns for analysis
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            # No numeric columns to scale
            report['steps_applied'].append('feature_scaling_skipped')
            report['scaling_method'] = 'None'
            report['scaling_reason'] = "No numeric features to scale"
            return X
        
        # Initialize scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Analyze each numeric column and apply appropriate scaling
        X_scaled = X.copy()
        scaling_methods = {}
        scaling_reasons = {}
        
        for col in numeric_columns:
            column_data = X[col].dropna()
            
            if len(column_data) == 0:
                continue
                
            # Check for outliers using IQR method
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((column_data < Q1 - 1.5 * IQR) | (column_data > Q3 + 1.5 * IQR)).sum()
            outlier_percentage = outliers / len(column_data)
            
            # Test for normal distribution using Shapiro-Wilk test
            try:
                _, p_value = stats.shapiro(column_data)
                is_normal = p_value > 0.05  # 5% significance level
            except:
                # If Shapiro-Wilk fails, use a simpler normality check
                is_normal = abs(stats.skew(column_data)) < 1 and abs(stats.kurtosis(column_data)) < 2
            
            # Determine scaling method based on distribution characteristics
            if outlier_percentage > 0.1:  # More than 10% outliers
                scaling_methods[col] = 'robust'
                scaling_reasons[col] = f"High outliers ({outlier_percentage:.1%})"
                # Apply robust scaling to this column
                robust_scaler = RobustScaler()
                X_scaled[col] = robust_scaler.fit_transform(X[[col]])
            elif is_normal:
                scaling_methods[col] = 'standard'
                scaling_reasons[col] = f"Normal distribution (p={p_value:.3f})"
                # Apply standard scaling to this column
                standard_scaler = StandardScaler()
                X_scaled[col] = standard_scaler.fit_transform(X[[col]])
            else:
                scaling_methods[col] = 'minmax'
                scaling_reasons[col] = f"Non-normal distribution (skew={stats.skew(column_data):.2f})"
                # Apply min-max scaling to this column
                minmax_scaler = MinMaxScaler()
                X_scaled[col] = minmax_scaler.fit_transform(X[[col]])
        
        # Update report
        report['steps_applied'].append('feature_scaling')
        report['scaling_method'] = 'Mixed'
        report['scaling_reason'] = f"Applied different scalers based on column characteristics: {dict(scaling_methods)}"
        report['scaling_details'] = {
            'methods': scaling_methods,
            'reasons': scaling_reasons
        }
        
        return X_scaled
    

    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str, report: Dict) -> pd.DataFrame:
        """
        Select the most important features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            problem_type (str): Problem type
            report (Dict): Engineering report to update
            
        Returns:
            pd.DataFrame: Selected features
        """
        try:
            # Clean data before feature selection
            X_clean = self._clean_data_for_selection(X, y)
            
            # Determine number of features to select (keep top 80% or max 20 features)
            n_features = min(int(X_clean.shape[1] * 0.8), 30, X_clean.shape[1])
            
            if problem_type == 'classification':
                if len(y.unique()) == 2:
                    # Binary classification
                    selector = SelectKBest(score_func=f_classif, k=n_features)
                else:
                    # Multi-class classification
                    selector = SelectKBest(score_func=f_classif, k=n_features)
            else:
                # Regression
                selector = SelectKBest(score_func=f_regression, k=n_features)
            
            # Fit the selector with error handling
            X_selected = selector.fit_transform(X_clean, y)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
            self.feature_selector = selector
            
            # Store feature importance scores with error handling
            try:
                feature_scores = pd.DataFrame({
                    'feature': X_clean.columns,
                    'score': selector.scores_,
                    'p_value': selector.pvalues_
                }).sort_values('score', ascending=False)
            except Exception as e:
                logging.warning(f"Error calculating feature scores: {str(e)}")
                # Create fallback feature scores
                feature_scores = pd.DataFrame({
                    'feature': X_clean.columns,
                    'score': np.ones(len(X_clean.columns)),
                    'p_value': np.ones(len(X_clean.columns))
                })
            
            report['feature_importance'] = feature_scores.to_dict('records')
            report['selected_features'] = selected_features
            report['steps_applied'].append('feature_selection')
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            # Return original features if feature selection fails
            report['steps_applied'].append('feature_selection_skipped')
            report['feature_importance'] = []
            report['selected_features'] = X.columns.tolist()
            return X
    
    def _reduce_dimensions(self, X: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """
        Apply dimensionality reduction if needed.
        
        Args:
            X (pd.DataFrame): Feature matrix
            report (Dict): Engineering report to update
            
        Returns:
            pd.DataFrame: Reduced features
        """
        # Only apply PCA if we have more than 10 features
        if X.shape[1] > 10:
            # Keep 95% of variance
            self.pca = PCA(n_components=0.95)
            X_reduced = self.pca.fit_transform(X)
            
            # Create new column names
            pca_columns = [f'PC_{i+1}' for i in range(X_reduced.shape[1])]
            
            report['pca_explained_variance'] = self.pca.explained_variance_ratio_.tolist()
            report['pca_components'] = X_reduced.shape[1]
            report['steps_applied'].append('dimensionality_reduction')
            
            return pd.DataFrame(X_reduced, columns=pca_columns, index=X.index)
        
        return X
    
    def _clean_data_for_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Clean data before feature selection to prevent warnings and errors.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Cleaned feature matrix
        """
        X_clean = X.copy()
        
        # Remove features with zero variance
        zero_var_features = X_clean.columns[X_clean.var() == 0]
        if len(zero_var_features) > 0:
            X_clean = X_clean.drop(columns=zero_var_features)
            logging.info(f"Removed {len(zero_var_features)} features with zero variance")
        
        # Remove features with infinite values
        inf_features = X_clean.columns[np.isinf(X_clean).any()]
        if len(inf_features) > 0:
            X_clean = X_clean.drop(columns=inf_features)
            logging.info(f"Removed {len(inf_features)} features with infinite values")
        
        # Replace NaN values with median for remaining features
        numeric_features = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if not np.isnan(median_val):
                    X_clean[col] = X_clean[col].fillna(median_val)
                else:
                    # If median is also NaN, drop the column
                    X_clean = X_clean.drop(columns=[col])
                    logging.info(f"Removed feature {col} due to all NaN values")
        
        # Ensure no infinite values remain
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Ensure target has no NaN values
        if y.isna().any():
            y_clean = y.fillna(y.mode()[0] if len(y.mode()) > 0 else y.median())
        else:
            y_clean = y
        
        return X_clean
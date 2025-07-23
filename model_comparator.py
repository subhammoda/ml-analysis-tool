import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import logging
from utils import detect_problem_type

class ModelComparator:
    """
    A class to compare multiple machine learning models and evaluate their performance.
    """
    
    def __init__(self):
        """Initialize the ModelComparator with default settings."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = None
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series, problem_type: str = 'auto') -> Dict:
        """
        Compare multiple models and return performance metrics.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            problem_type (str): 'classification', 'regression', or 'auto'
            
        Returns:
            Dict: Comparison results with performance metrics
        """
        logging.info("Starting model comparison")
        logging.info(f"  Input shape: {X.shape}")
        logging.info(f"  Target variable: {y.name}")
        
        if problem_type == 'auto':
            problem_type = self._detect_problem_type(y)
            logging.info(f"  Detected problem type: {problem_type}")
        
        # Initialize models based on problem type
        logging.info("  Initializing models")
        self._initialize_models(problem_type)
        logging.info(f"  Initialized {len(self.models)} models")
        
        # Split data
        logging.info("  Splitting data")
        if problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info("  Data split with stratification")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("  Data split completed")
        
        logging.info(f"  Training set: {X_train.shape[0]} samples")
        logging.info(f"  Test set: {X_test.shape[0]} samples")
        
        # Train and evaluate models
        logging.info("  Training and evaluating models")
        comparison_results = self._evaluate_models(X_train, X_test, y_train, y_test, problem_type)
        
        # Find best model
        logging.info("  Finding best model")
        self._find_best_model(comparison_results, problem_type)
        
        # Generate detailed analysis
        logging.info("  Generating detailed analysis")
        detailed_analysis = self._generate_detailed_analysis(X_train, X_test, y_train, y_test, problem_type)
        
        logging.info(f"  Models evaluated: {len(comparison_results)}")
        logging.info(f"  Best model: {self.best_model}")
        logging.info(f"  Best score: {self.best_score:.4f}")
        logging.info(f"Model comparison completed")
        
        return {
            'comparison_results': comparison_results,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'detailed_analysis': detailed_analysis,
            'problem_type': problem_type
        }
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """
        Automatically detect if the problem is classification or regression.
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            str: 'classification' or 'regression'
        """
        return detect_problem_type(y)
    
    def _initialize_models(self, problem_type: str) -> None:
        """
        Initialize models based on problem type.
        
        Args:
            problem_type (str): Type of problem
        """
        if problem_type == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Naive Bayes': GaussianNB()
            }
        else:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
    
    def _evaluate_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series, problem_type: str) -> Dict:
        """
        Evaluate all models and return performance metrics.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            problem_type (str): Type of problem
            
        Returns:
            Dict: Performance metrics for all models
        """
        results = {}
        successful_models = 0
        failed_models = 0
        
        logging.info(f"  Evaluating {len(self.models)} models:")
        
        for i, (name, model) in enumerate(self.models.items(), 1):
            logging.info(f"    [{i}/{len(self.models)}] Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, problem_type)
                
                # Cross-validation score
                cv_score = self._cross_validate_model(model, X_train, y_train, problem_type)
                
                # Log performance
                if problem_type == 'classification':
                    score = metrics.get('f1_score', 0)
                    metric_name = 'F1 Score'
                else:
                    score = metrics.get('r2_score', 0)
                    metric_name = 'R² Score'
                
                logging.info(f"    {name}: {metric_name} = {score:.4f}")
                
                results[name] = {
                    'metrics': metrics,
                    'cv_score': cv_score,
                    'predictions': y_pred.tolist(),
                    'model': model
                }
                successful_models += 1
                
            except Exception as e:
                logging.error(f"   {name}: {str(e)}")
                results[name] = {
                    'error': str(e),
                    'metrics': {},
                    'cv_score': None
                }
                failed_models += 1
        
        logging.info(f"  Model evaluation summary:")
        logging.info(f"  Successful: {successful_models}")
        logging.info(f"  Failed: {failed_models}")
        
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray], problem_type: str) -> Dict:
        """
        Calculate performance metrics based on problem type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities (for classification)
            problem_type: Type of problem
            
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
        else:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Dict:
        """
        Perform cross-validation for a model.
        
        Args:
            model: The model to validate
            X, y: Training data
            problem_type: Type of problem
            
        Returns:
            Dict: Cross-validation results
        """
        try:
            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                # Use F1 score for multi-class, accuracy for binary
                if len(np.unique(y)) == 2:
                    scoring = 'accuracy'
                else:
                    scoring = 'f1_weighted'
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scoring = 'r2'
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return {
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std()),
                'scores': cv_scores.tolist()
            }
        
        except Exception as e:
            logging.error(f"Error in cross-validation: {str(e)}")
            return {'error': str(e)}
    
    def _find_best_model(self, results: Dict, problem_type: str) -> None:
        """
        Find the best performing model.
        
        Args:
            results: Model comparison results
            problem_type: Type of problem
        """
        best_score = float('-inf') if problem_type == 'classification' else float('-inf')
        best_model_name = None
        
        for name, result in results.items():
            if 'error' in result:
                continue
            
            if problem_type == 'classification':
                # Use F1 score as primary metric
                score = result['metrics'].get('f1_score', 0)
                if score > best_score:
                    best_score = score
                    best_model_name = name
            else:
                # Use R² score as primary metric for regression
                score = result['metrics'].get('r2_score', float('-inf'))
                if score > best_score:
                    best_score = score
                    best_model_name = name
        
        self.best_model = best_model_name
        self.best_score = best_score
    
    def _generate_detailed_analysis(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                  y_train: pd.Series, y_test: pd.Series, problem_type: str) -> Dict:
        """
        Generate detailed analysis of the best model.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            problem_type: Type of problem
            
        Returns:
            Dict: Detailed analysis results
        """
        if not self.best_model or self.best_model not in self.models:
            return {'error': 'No best model found'}
        
        best_model = self.models[self.best_model]
        
        # Retrain best model on full training data
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        analysis = {
            'model_name': self.best_model,
            'best_score': self.best_score,
            'feature_importance': self._get_feature_importance(best_model, X_train.columns),
            'prediction_analysis': self._analyze_predictions(y_test, y_pred, problem_type)
        }
        
        if problem_type == 'classification':
            analysis['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return analysis
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """
        Get feature importance from the model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dict: Feature importance scores
        """
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = dict(zip(feature_names, np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def _analyze_predictions(self, y_true: pd.Series, y_pred: np.ndarray, problem_type: str) -> Dict:
        """
        Analyze prediction errors and patterns.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            problem_type: Type of problem
            
        Returns:
            Dict: Prediction analysis
        """
        analysis = {}
        
        if problem_type == 'regression':
            errors = y_true - y_pred
            analysis['error_statistics'] = {
                'mean_error': float(errors.mean()),
                'std_error': float(errors.std()),
                'min_error': float(errors.min()),
                'max_error': float(errors.max())
            }
            
            # Error distribution
            analysis['error_distribution'] = {
                'positive_errors': int((errors > 0).sum()),
                'negative_errors': int((errors < 0).sum()),
                'zero_errors': int((errors == 0).sum())
            }
        
        return analysis
    
 
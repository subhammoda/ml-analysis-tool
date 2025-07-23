import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from model_comparator import ModelComparator
from llm_recommender import LLMRecommender
from utils import detect_problem_type

class MLAnalyzer:
    """
    Main class that orchestrates the entire machine learning analysis pipeline.
    """
    
    def __init__(self, data_processor: DataProcessor, feature_engineer: FeatureEngineer, 
                 model_comparator: ModelComparator):
        """
        Initialize the ML Analyzer with its components.
        
        Args:
            data_processor: DataProcessor instance
            feature_engineer: FeatureEngineer instance
            model_comparator: ModelComparator instance
        """
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.model_comparator = model_comparator
        self.llm_recommender = LLMRecommender()
        self.analysis_results = {}
    
    def perform_analysis(self, target_column: str, data_quality_report: Dict = None, drop_columns: Optional[List[str]] = None) -> Dict:
        """
        Perform complete machine learning analysis.
        
        Args:
            target_column (str): Name of the target column
            drop_columns (Optional[List[str]]): List of columns to drop from features
            
        Returns:
            Dict: Complete analysis results
        """
        logging.info("=" * 60)
        logging.info("STARTING MACHINE LEARNING ANALYSIS")
        logging.info("=" * 60)
        logging.info(f"Target column: {target_column}")
        
        try:
            # Step 1: Data Quality Analysis
            logging.info("Step 1: Data Quality Analysis")
            if data_quality_report is None:
                data_quality_report = self._analyze_data_quality()
            logging.info(f"Data quality analysis completed")
            
            # Step 2: Data Preprocessing
            logging.info("Step 2: Data Preprocessing")
            X, y = self.data_processor.preprocess_data(target_column, drop_columns=drop_columns)
            
            # Step 3: Feature Engineering
            logging.info("Step 3: Feature Engineering")
            X_engineered, feature_engineering_report = self.feature_engineer.engineer_features(X, y)
            
            # Step 4: Model Comparison
            logging.info("Step 4: Model Comparison")
            model_comparison_results = self.model_comparator.compare_models(X_engineered, y)
            problem_type = model_comparison_results.get('problem_type', 'unknown')
            best_model = model_comparison_results.get('best_model', 'unknown')
            best_score = model_comparison_results.get('best_score', 0)
            
            # Step 5: Feature Importance Analysis
            logging.info("Step 5: Feature Importance Analysis")
            feature_importance_analysis = self._analyze_feature_importance(X, y)
            logging.info(f"Feature importance analysis completed")
            
            # Step 6: Generate LLM-based Recommendations
            logging.info("Step 6: Generating AI Insights")
            recommendations = self._generate_llm_recommendations(model_comparison_results, feature_engineering_report, data_quality_report)
            logging.info(f"AI insights generation completed")
            
            # Step 7: Compile Results
            logging.info("Step 7: Compiling Results")
            self.analysis_results = {
                'data_quality': data_quality_report,
                'feature_engineering': feature_engineering_report,
                'model_comparison': model_comparison_results,
                'feature_importance': feature_importance_analysis,
                'recommendations': recommendations,
                'summary': self._generate_summary(model_comparison_results, feature_engineering_report)
            }
            
            logging.info("=" * 60)
            logging.info("ANALYSIS COMPLETED SUCCESSFULLY")
            logging.info("=" * 60)
            logging.info(f"📊 Summary:")
            logging.info(f"- Problem Type: {problem_type}")
            logging.info(f"- Best Model: {best_model}")
            logging.info(f"- Best Score: {best_score:.4f}")
            logging.info(f"- Feature Reduction: {feature_engineering_report.get('feature_reduction', 0)}")
            logging.info(f"- Models Evaluated: {len(model_comparison_results.get('comparison_results', {}))}")
            logging.info("=" * 60)
            
            return self.analysis_results
        
        except Exception as e:
            logging.error("=" * 60)
            logging.error("❌ ANALYSIS FAILED")
            logging.error("=" * 60)
            logging.error(f"Error in ML analysis: {str(e)}")
            logging.error("=" * 60)
            return {'error': str(e)}
    
    def _analyze_data_quality(self) -> Dict:
        """
        Analyze data quality and generate report.
        
        Returns:
            Dict: Data quality analysis
        """
        return self.data_processor.get_data_quality_report()
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform feature importance analysis using preprocessed data.
        
        Args:
            X: Preprocessed feature dataframe
            y: Preprocessed target variable
            
        Returns:
            Dict: Feature importance analysis results
        """
        try:
            # Use Random Forest for feature importance
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Determine if it's classification or regression
            problem_type = detect_problem_type(y)
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                "feature_importance": feature_importance.to_dict('records'),
                "model_type": problem_type,
                "target_distribution": y.value_counts().to_dict() if problem_type == 'classification' else {
                    "min": float(y.min()),
                    "max": float(y.max()),
                    "mean": float(y.mean()),
                    "std": float(y.std())
                }
            }
        
        except Exception as e:
            logging.error(f"  Error in feature importance analysis: {str(e)}")
            return {"error": str(e)}
    
    def _generate_llm_recommendations(self, model_comparison_results: Dict, 
                                    feature_engineering_report: Dict,
                                    data_quality_report: Dict) -> Dict:
        """
        Generate LLM-based recommendations and insights.
        
        Args:
            model_comparison_results: Results from model comparison
            feature_engineering_report: Feature engineering report
            data_quality_report: Data quality report
            
        Returns:
            Dict: LLM-generated recommendations and insights
        """
        try:
            # Get sample data from data processor
            data_info = self.data_processor.get_data_info()
            sample_data = data_info.get('sample_data', [])
            
            # Generate comprehensive insights using LLM
            llm_insights = self.llm_recommender.generate_model_insights(
                model_comparison_results,
                feature_engineering_report,
                data_quality_report,
                sample_data
            )
            
            return {
                'llm_insights': llm_insights,
                'source': 'LLM (Gemini)',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error generating LLM recommendations: {str(e)}")
            return {
                'error': f"Failed to generate LLM recommendations: {str(e)}",
                'fallback_recommendations': self._generate_fallback_recommendations(model_comparison_results)
            }
    
    def _generate_fallback_recommendations(self, model_comparison_results: Dict) -> Dict:
        """
        Generate fallback recommendations if LLM fails.
        
        Args:
            model_comparison_results: Results from model comparison
            
        Returns:
            Dict: Basic fallback recommendations
        """
        problem_type = model_comparison_results.get('problem_type', 'unknown')
        best_model = model_comparison_results.get('best_model', 'unknown')
        best_score = model_comparison_results.get('best_score', 0)
        
        recommendations = {
            'problem_type': problem_type,
            'best_model': best_model,
            'best_score': best_score,
            'recommendations': [
                "LLM analysis failed - using fallback recommendations",
                f"Best performing model: {best_model}",
                f"Best score: {best_score:.4f}",
                "Consider hyperparameter tuning for the best model",
                "Implement cross-validation for robust evaluation",
                "Set up model monitoring and alerting"
            ]
        }
        
        return recommendations
    
    def _generate_summary(self, model_comparison_results: Dict, feature_engineering_report: Dict) -> Dict:
        """
        Generate a comprehensive summary of the analysis.
        
        Args:
            model_comparison_results: Results from model comparison
            feature_engineering_report: Feature engineering report
            
        Returns:
            Dict: Analysis summary
        """
        best_model = model_comparison_results.get('best_model')
        best_score = model_comparison_results.get('best_score')
        problem_type = model_comparison_results.get('problem_type')
        
        summary = {
            'best_model': best_model,
            'best_score': best_score,
            'problem_type': problem_type,
            'feature_reduction': feature_engineering_report.get('feature_reduction', 0),
            'total_models_evaluated': len(model_comparison_results.get('comparison_results', {})),
            'performance_insights': self._get_performance_insights(model_comparison_results, problem_type)
        }
        
        return summary
    
    def _get_performance_insights(self, model_comparison_results: Dict, problem_type: str) -> List[str]:
        """
        Generate performance insights based on model comparison results.
        
        Args:
            model_comparison_results: Results from model comparison
            problem_type: Type of problem
            
        Returns:
            List[str]: Performance insights
        """
        insights = []
        results = model_comparison_results.get('comparison_results', {})
        
        if not results:
            return ["No models were successfully evaluated"]
        
        # Analyze model performance distribution
        scores = []
        for name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                if problem_type == 'classification':
                    score = result['metrics'].get('f1_score', 0)
                else:
                    score = result['metrics'].get('r2_score', 0)
                scores.append((name, score))
        
        if scores:
            scores.sort(key=lambda x: x[1], reverse=(problem_type == 'classification'))
            
            best_score = scores[0][1]
            worst_score = scores[-1][1]
            score_range = best_score - worst_score
            
            insights.append(f"Best performing model: {scores[0][0]} with score {best_score:.4f}")
            insights.append(f"Score range across models: {score_range:.4f}")
            
            if score_range < 0.1:
                insights.append("Models show similar performance - consider ensemble methods")
            elif score_range > 0.3:
                insights.append("High variance in model performance - focus on best model")
        
        return insights
import google.generativeai as genai
import os
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMRecommender:
    """
    A class to generate intelligent recommendations using LLM based on model comparison results.
    """
    
    def __init__(self):
        """Initialize the LLM Recommender with Gemini configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate_model_insights(self, model_comparison_results: Dict, 
                               feature_engineering_report: Dict,
                               data_quality_report: Dict,
                               sample_data: List[Dict] = None) -> Dict:
        """
        Generate comprehensive insights and recommendations using LLM.
        
        Args:
            model_comparison_results: Results from model comparison
            feature_engineering_report: Feature engineering report
            data_quality_report: Data quality report
            
        Returns:
            Dict: LLM-generated insights and recommendations
        """
        logging.info("Starting LLM insights generation")
        
        try:
            # Prepare the data for LLM analysis
            logging.info("  Preparing analysis data")
            analysis_data = self._prepare_analysis_data(
                model_comparison_results, 
                feature_engineering_report, 
                data_quality_report,
                sample_data
            )
            logging.info("  Analysis data prepared")
            
            # Generate insights using LLM
            logging.info("  Generating LLM insights")
            insights = self._generate_llm_insights(analysis_data)
            logging.info("  LLM insights generated successfully")
            
            return insights
            
        except Exception as e:
            logging.error(f"  Error generating LLM insights: {str(e)}")
            return {
                'error': f"Failed to generate LLM insights: {str(e)}",
                'fallback_recommendations': self._generate_fallback_recommendations(model_comparison_results)
            }
    
    def _prepare_analysis_data(self, model_comparison_results: Dict, 
                              feature_engineering_report: Dict,
                              data_quality_report: Dict,
                              sample_data: List[Dict] = None) -> Dict:
        """
        Prepare analysis data for LLM consumption.
        
        Args:
            model_comparison_results: Model comparison results
            feature_engineering_report: Feature engineering report
            data_quality_report: Data quality report
            
        Returns:
            Dict: Formatted analysis data
        """
        # Extract key information from model comparison
        comparison_results = model_comparison_results.get('comparison_results', {})
        problem_type = model_comparison_results.get('problem_type', 'unknown')
        best_model = model_comparison_results.get('best_model', 'unknown')
        best_score = model_comparison_results.get('best_score', 0)
        
        # Prepare model performance data
        model_performance = []
        for model_name, result in comparison_results.items():
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                cv_score = result.get('cv_score', {})
                
                model_data = {
                    'model_name': model_name,
                    'metrics': metrics,
                    'cv_mean': cv_score.get('mean', 0) if cv_score else 0,
                    'cv_std': cv_score.get('std', 0) if cv_score else 0
                }
                model_performance.append(model_data)
        
        # Prepare feature engineering data
        fe_data = {
            'original_features': feature_engineering_report.get('original_features', 0),
            'final_features': feature_engineering_report.get('final_features', 0),
            'feature_reduction': feature_engineering_report.get('feature_reduction', 0),
            'scaling_method': feature_engineering_report.get('scaling_method', 'unknown'),
            'scaling_reason': feature_engineering_report.get('scaling_reason', ''),
            'steps_applied': feature_engineering_report.get('steps_applied', [])
        }
        
        # Prepare data quality data
        dq_data = {
            'total_rows': data_quality_report.get('total_rows', 0),
            'total_columns': data_quality_report.get('total_columns', 0),
            'missing_data_percentage': data_quality_report.get('missing_data_percentage', 0),
            'duplicate_rows': data_quality_report.get('duplicate_rows', 0),
            'numeric_columns_count': data_quality_report.get('numeric_columns_count', 0),
            'categorical_columns_count': data_quality_report.get('categorical_columns_count', 0)
        }
        
        return {
            'problem_type': problem_type,
            'best_model': best_model,
            'best_score': best_score,
            'model_performance': model_performance,
            'feature_engineering': fe_data,
            'data_quality': dq_data,
            'sample_data': sample_data
        }
    
    def _generate_llm_insights(self, analysis_data: Dict) -> Dict:
        """
        Generate insights using LLM based on analysis data.
        
        Args:
            analysis_data: Prepared analysis data
            
        Returns:
            Dict: LLM-generated insights
        """
        # Create comprehensive prompt for LLM
        prompt = self._create_analysis_prompt(analysis_data)
        
        # Generate response from LLM
        response = self.model.generate_content(prompt)
        
        # Parse and structure the response
        insights = self._parse_llm_response(response.text)
        
        return insights
    
    def _create_analysis_prompt(self, analysis_data: Dict) -> str:
        """
        Create a comprehensive prompt for LLM analysis requesting JSON response.
        
        Args:
            analysis_data: Analysis data
            
        Returns:
            str: Formatted prompt
        """
        problem_type = analysis_data['problem_type']
        best_model = analysis_data['best_model']
        best_score = analysis_data['best_score']
        model_performance = analysis_data['model_performance']
        fe_data = analysis_data['feature_engineering']
        dq_data = analysis_data['data_quality']
        sample_data = analysis_data.get('sample_data', [])
        
        # Format model performance data
        model_summary = []
        for model in model_performance:
            metrics = model['metrics']
            if problem_type == 'classification':
                score = metrics.get('f1_score', 0)
                metric_name = 'F1 Score'
            else:
                score = metrics.get('r2_score', 0)
                metric_name = 'R² Score'
            
            model_summary.append(f"- {model['model_name']}: {score:.4f} ({metric_name})")
        
        prompt = f"""
        You are an expert machine learning consultant. Analyze the following ML analysis results and provide detailed, actionable recommendations with specific technical guidance.

        ANALYSIS DATA:
        
        Problem Type: {problem_type}
        Best Model: {best_model}
        Best Score: {best_score:.4f}
        
        Model Performance Summary:
        {chr(10).join(model_summary)}
        
        Feature Engineering:
        - Original Features: {fe_data['original_features']}
        - Final Features: {fe_data['final_features']}
        - Feature Reduction: {fe_data['feature_reduction']}
        - Scaling Method: {fe_data['scaling_method']}
        - Scaling Reason: {fe_data['scaling_reason']}
        - Steps Applied: {', '.join(fe_data['steps_applied'])}
        
        Data Quality:
        - Total Rows: {dq_data['total_rows']}
        - Total Columns: {dq_data['total_columns']}
        - Missing Data: {dq_data['missing_data_percentage']:.2f}%
        - Duplicate Rows: {dq_data['duplicate_rows']}
        - Numeric Columns: {dq_data['numeric_columns_count']}
        - Categorical Columns: {dq_data['categorical_columns_count']}
        
        Sample Data (First 10 rows):
        {json.dumps(sample_data, indent=2) if sample_data else "No sample data available"}
        
        Please provide detailed recommendations in the following JSON format. For each list field, provide ONLY the TOP 2 most important suggestions with specific technical details:
        
        {{
            "recommendations": {{
                "model_improvements": [
                    "Detailed model improvement with specific hyperparameter tuning suggestions, ensemble methods, or advanced techniques",
                    "Second detailed model enhancement with specific implementation guidance"
                ],
                "feature_engineering_enhancements": [
                    "Detailed feature engineering improvement with specific techniques and implementation details",
                    "Second detailed feature enhancement with specific guidance"
                ],
                "data_collection_suggestions": [
                    "Detailed data collection/cleaning suggestion with specific methods and tools",
                    "Second detailed data improvement with specific implementation steps"
                ]
            }}
        }}
        
        IMPORTANT GUIDELINES: 
        - Respond ONLY with valid JSON. Do not include any text before or after the JSON object.
        - For model improvements, include specific hyperparameter ranges, ensemble methods, or advanced techniques appropriate for the best model.
        - For feature engineering, suggest specific techniques like polynomial features, interaction terms, domain-specific features, or advanced encoding methods.
        - For data collection, suggest specific data augmentation techniques, additional data sources, or cleaning methods.

        - Provide actionable, technical details that can be directly implemented.
        - For all list fields, provide ONLY the TOP 2 most important suggestions with comprehensive technical guidance.
        """
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse and structure the LLM JSON response.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Dict: Structured insights
        """
        try:
            # Clean the response text to extract JSON
            cleaned_response = self._clean_json_response(response_text)
            
            # Parse JSON response
            parsed_response = json.loads(cleaned_response)
            
            # Add metadata
            parsed_response['source'] = 'llm_analysis'
            parsed_response['response_format'] = 'structured_json'
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM JSON response: {str(e)}")
            logging.error(f"Raw response: {response_text}")
            
            # Return fallback structure with raw response
            return {
                'error': f"Failed to parse JSON response: {str(e)}",
                'raw_response': response_text,
                'fallback_analysis': {
                    'performance_analysis': 'LLM response parsing failed',
                    'recommendations': ['Review the raw LLM response for insights'],
                    'summary': 'JSON parsing failed - check raw response'
                }
            }
        except Exception as e:
            logging.error(f"Unexpected error parsing LLM response: {str(e)}")
            return {
                'error': f"Unexpected error: {str(e)}",
                'raw_response': response_text,
                'fallback_analysis': {
                    'performance_analysis': 'LLM response processing failed',
                    'recommendations': ['Review the raw LLM response for insights'],
                    'summary': 'Response processing failed - check raw response'
                }
            }
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean the LLM response to extract valid JSON.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove common LLM response artifacts
        cleaned = response_text.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Find the first { and last } to extract JSON
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx + 1]
        
        return cleaned
    
    def _generate_fallback_recommendations(self, model_comparison_results: Dict) -> Dict:
        """
        Generate fallback recommendations if LLM fails.
        
        Args:
            model_comparison_results: Model comparison results
            
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
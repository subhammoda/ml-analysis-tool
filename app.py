import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_analysis.log')
    ]
)

# Import our ML classes
from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from model_comparator import ModelComparator
from ml_analyzer import MLAnalyzer

# Page configuration
st.set_page_config(
    page_title="ML Analysis Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode
st.markdown("""
<style>
    .main {
        background-color: #0f0f23;
        color: #e0e0e0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    .stSelectbox, .stTextInput, .stTextArea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #e0e0e0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #64b5f6, #42a5f5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #42a5f5, #2196f3);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(100, 181, 246, 0.3);
    }

    .footer {
        text-align: center;
        color: #718096;
        font-style: italic;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🧠 ML Analysis Tool")
    st.markdown("Upload your dataset, specify the target column, and get comprehensive machine learning analysis")
    
    # Initialize session state
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
        
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "🔧 Analysis", "📈 Results", "🤖 AI Insights"])
    
    with tab1:

        col_quick1, col_or, col_quick2 = st.columns([1, 0.1, 1])
    
        with col_quick1:
            st.subheader("🚀 Sample Dataset")
            sample_files = [f for f in os.listdir('sample_data') if f.endswith('.csv')]
            sample_choice = st.selectbox(
                "Or select a sample dataset:",
                options=[None] + sample_files,
                format_func=lambda x: x if x else "-- Select a sample dataset --",
                help="Choose a sample dataset to explore the tool without uploading your own file."
            )
        
        with col_or:
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: center;">
                <div style="border-left: 2px solid #ccc; height: 60px; margin: 10px 0; display: block;" class="desktop-line"></div>
                <div style="font-weight: bold; color: #666; margin: 5px 0; font-size: 18px;">OR</div>
                <div style="border-left: 2px solid #ccc; height: 60px; margin: 10px 0; display: block;" class="desktop-line"></div>
            </div>
            <style>
            @media (max-width: 768px) {
                .desktop-line {
                    display: none !important;
                }
            }
            </style>
            """, unsafe_allow_html=True)
        
        with col_quick2:
            # File upload section
            st.subheader("📁 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your structured data file"
            )
        
        # Mobile responsive separator
        st.markdown("""
        <style>
        @media (max-width: 768px) {
            .mobile-separator {
                display: block !important;
                text-align: center;
                margin: 20px 0;
            }
            .mobile-separator::before {
                content: '';
                display: inline-block;
                width: 50px;
                height: 2px;
                background-color: #ccc;
                margin: 0 10px;
                vertical-align: middle;
            }
            .mobile-separator::after {
                content: '';
                display: inline-block;
                width: 50px;
                height: 2px;
                background-color: #ccc;
                margin: 0 10px;
                vertical-align: middle;
            }
        }
        @media (min-width: 769px) {
            .mobile-separator {
                display: none !important;
            }
        }
        </style>
        <div class="mobile-separator" style="display: none;">
            <span style="font-weight: bold; color: #666;">OR</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Only allow one input method at a time
        if uploaded_file is not None and sample_choice:
            st.warning("Please use either file upload or sample dataset, not both.")
        elif sample_choice:
            try:
                sample_path = os.path.join('sample_data', sample_choice)
                st.session_state.data_processor = DataProcessor(sample_path)
                st.session_state.current_file = sample_path
                data_info = st.session_state.data_processor.get_data_info()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", data_info['shape'][0])
                with col2:
                    st.metric("Columns", data_info['shape'][1])
                with col3:
                    st.metric("Memory Usage", f"{data_info.get('memory_usage_mb', 0):.2f} MB")
                st.subheader("📋 Sample Data")
                sample_df = pd.DataFrame(data_info['sample_data'])
                st.dataframe(sample_df, use_container_width=True)
                quality_report = st.session_state.data_processor.get_data_quality_report()
                st.session_state.data_quality_report = quality_report
                st.subheader("📊 Data Quality Report")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Missing Data %", f"{quality_report['missing_data_percentage']:.2f}%")
                with col2:
                    st.metric("Duplicate Rows", quality_report['duplicate_rows'])
                with col3:
                    st.metric("Numeric Columns", quality_report['numeric_columns_count'])
                with col4:
                    st.metric("Categorical Columns", quality_report['categorical_columns_count'])
                if quality_report.get('high_missing_count', 0) > 0:
                    st.warning(f"⚠️ {quality_report['high_missing_count']} column(s) with >50% missing data will be dropped during analysis")
                    if quality_report.get('columns_with_high_missing'):
                        st.write("**Columns to be dropped:**")
                        for col in quality_report['columns_with_high_missing']:
                            st.write(f"• {col}")
                st.success("✅ Sample dataset loaded and processed successfully!")
            except Exception as e:
                st.error(f"❌ Error loading sample dataset: {str(e)}")
        elif uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                # Process the file
                st.session_state.data_processor = DataProcessor(tmp_file_path)
                st.session_state.current_file = tmp_file_path
                # Display data info
                data_info = st.session_state.data_processor.get_data_info()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", data_info['shape'][0])
                with col2:
                    st.metric("Columns", data_info['shape'][1])
                with col3:
                    st.metric("Memory Usage", f"{data_info.get('memory_usage_mb', 0):.2f} MB")
                # Show sample data
                st.subheader("📋 Sample Data")
                sample_df = pd.DataFrame(data_info['sample_data'])
                st.dataframe(sample_df, use_container_width=True)
                # Data quality report
                quality_report = st.session_state.data_processor.get_data_quality_report()
                st.session_state.data_quality_report = quality_report
                
                st.subheader("📊 Data Quality Report")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Missing Data %", f"{quality_report['missing_data_percentage']:.2f}%")
                with col2:
                    st.metric("Duplicate Rows", quality_report['duplicate_rows'])
                with col3:
                    st.metric("Numeric Columns", quality_report['numeric_columns_count'])
                with col4:
                    st.metric("Categorical Columns", quality_report['categorical_columns_count'])
                
                # Show information about columns with high missing data
                if quality_report.get('high_missing_count', 0) > 0:
                    st.warning(f"⚠️ {quality_report['high_missing_count']} column(s) with >50% missing data will be dropped during analysis")
                    if quality_report.get('columns_with_high_missing'):
                        st.write("**Columns to be dropped:**")
                        for col in quality_report['columns_with_high_missing']:
                            st.write(f"• {col}")
                
                st.success("✅ Dataset uploaded and processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab2:
        st.header("🔧 Configure Analysis")
        
        if st.session_state.data_processor is None:
            st.warning("⚠️ Please upload a dataset first in the 'Upload Data' tab.")
        else:
            data_info = st.session_state.data_processor.get_data_info()
            
            # Column drop selection
            st.subheader("🗑️ Drop Columns from Model Training")
            drop_options = data_info['columns']
            drop_columns = st.multiselect(
                "Select columns to exclude from model training:",
                options=drop_options,
                help="You can drop columns like IDs, names, or any irrelevant features."
            )
            st.session_state['drop_columns'] = drop_columns

            # Target column selection (exclude dropped columns)
            st.subheader("🎯 Select Target Column")
            target_options = [col for col in data_info['columns'] if col not in drop_columns]
            target_column = st.selectbox(
                "Choose the column you want to predict:",
                options=target_options,
                help="This column will be used as the target variable for your ML models"
            )
            
            # Show analysis status
            if st.session_state.analysis_running:
                st.info("🔄 Analysis is currently running... Please wait.")
                
                # Progress bar
                progress_bar = st.progress(0)
                
                with st.spinner("🔄 Running analysis... This may take a few minutes..."):
                    try:
                        # Initialize components
                        progress_bar.progress(10)
                        
                        processor = st.session_state.data_processor
                        feature_engineer = FeatureEngineer()
                        model_comparator = ModelComparator()
                        analyzer = MLAnalyzer(processor, feature_engineer, model_comparator)
                        
                        progress_bar.progress(50)
                        
                        # Perform analysis
                        results = analyzer.perform_analysis(target_column, st.session_state.data_quality_report, drop_columns=st.session_state.get('drop_columns', []))
                        
                        progress_bar.progress(90)
                        
                        if 'error' not in results:
                            st.session_state.analysis_results = results
                            st.session_state.analysis_running = False
                            st.session_state.analysis_completed = True
                            progress_bar.progress(100)
                            st.success("✅ Analysis completed successfully!")
                            st.rerun()
                        else:
                            st.session_state.analysis_running = False
                            st.error(f"❌ Analysis failed: {results['error']}")
                            
                    except Exception as e:
                        st.session_state.analysis_running = False
                        st.error(f"❌ Error during analysis: {str(e)}")
            
            elif st.session_state.analysis_completed:
                st.success("✅ Analysis completed successfully!")
                st.info("📈 Check the 'Results' tab to view your analysis results.")
                
                # Reset completion state
                st.session_state.analysis_completed = False
            
            else:
                # Show start analysis button
                if st.button("🚀 Start Analysis", type="primary"):
                    if target_column:
                        # Set analysis running state
                        st.session_state.analysis_running = True
                        st.session_state.analysis_completed = False
                        st.success("✅ Analysis started! Please switch to the 'Analysis' tab to monitor progress.")
                        st.rerun()
                    else:
                        st.warning("⚠️ Please select a target column.")
    
    with tab3:
        st.header("📈 Analysis Results")
        
        if st.session_state.analysis_results is None:
            st.info("ℹ️ No analysis results available. Please run an analysis first.")
        else:
            results = st.session_state.analysis_results
            
            # Summary metrics
            if 'summary' in results:
                summary = results['summary']
                st.subheader("🏆 Best Model Summary")
                
                col1, col2, col3, col4 = st.columns([2,1,1,1])
                with col1:
                    st.metric("Best Model", summary.get('best_model', 'N/A'))
                with col2:
                    st.metric("Best Score", f"{summary.get('best_score', 0):.4f}")
                with col3:
                    st.metric("Problem Type", summary.get('problem_type', 'N/A'))
                with col4:
                    st.metric("Models Evaluated", summary.get('total_models_evaluated', 0))
            
            # Model comparison
            if 'model_comparison' in results and 'comparison_results' in results['model_comparison']:
                st.subheader("📊 Model Performance Comparison")
                
                model_results = results['model_comparison']['comparison_results']
                problem_type = results['model_comparison'].get('problem_type', 'classification')
                
                # Create comparison dataframe with problem-specific metrics
                comparison_data = []
                for model_name, result in model_results.items():
                    if 'error' not in result and 'metrics' in result:
                        metrics = result['metrics']
                        model_data = {'Model': model_name}
                        
                        # Add metrics based on problem type
                        if problem_type == 'classification':
                            model_data.update({
                                'Accuracy': metrics.get('accuracy', None),
                                'Precision': metrics.get('precision', None),
                                'Recall': metrics.get('recall', None),
                                'F1 Score': metrics.get('f1_score', None)
                            })
                        else:  # regression
                            model_data.update({
                                'RMSE': metrics.get('rmse', None),
                                'MAE': metrics.get('mae', None),
                                'R² Score': metrics.get('r2_score', None)
                            })
                        
                        comparison_data.append(model_data)
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Create performance chart
                    if len(comparison_data) > 0:
                        # Determine primary metric based on problem type
                        if problem_type == 'classification':
                            metric_col = 'F1 Score'
                            chart_title = 'Model Performance Comparison (F1 Score)'
                        else:
                            metric_col = 'R² Score'
                            chart_title = 'Model Performance Comparison (R² Score)'
                        
                        # Filter out None values
                        valid_data = df_comparison[df_comparison[metric_col].notna()]
                        
                        if not valid_data.empty:
                            fig = px.bar(
                                valid_data,
                                x='Model',
                                y=metric_col,
                                title=chart_title,
                                color=metric_col,
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#e0e0e0')
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if 'feature_engineering' in results and 'feature_importance' in results['feature_engineering']:
                st.subheader("⭐ Feature Importance")
                
                feature_importance = results['feature_engineering']['feature_importance']
                if feature_importance:
                    # Show only the selected features from feature engineering
                    fig = px.bar(
                        x=[item['score'] for item in feature_importance],
                        y=[item['feature'] for item in feature_importance],
                        orientation='h',
                        title='Selected Features - Feature Importance',
                        labels={'x': 'Score', 'y': 'Feature'}
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature count
                    st.info(f"📊 Total selected features: {len(feature_importance)}")
            
            # Feature engineering report
            if 'feature_engineering' in results:
                st.subheader("🔧 Feature Engineering Report")
                fe_report = results['feature_engineering']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Features", fe_report.get('original_features', 0))
                with col2:
                    st.metric("After Encoding", fe_report.get('features_after_encoding', 0))
                with col3:
                    st.metric("Selected Features", fe_report.get('final_features', 0))
                with col4:
                    st.metric("Total Reduction", fe_report.get('feature_reduction', 0))
                
                # Display scaling details if available
                if 'scaling_details' in fe_report:
                    st.subheader("📊 Scaling Details")
                    scaling_details = fe_report['scaling_details']
                    
                    if 'methods' in scaling_details and 'reasons' in scaling_details:
                        scaling_data = []
                        for col, method in scaling_details['methods'].items():
                            reason = scaling_details['reasons'].get(col, '')
                            scaling_data.append({
                                'Column': col,
                                'Method': method.upper(),
                                'Reason': reason
                            })
                        
                        if scaling_data:
                            scaling_df = pd.DataFrame(scaling_data)
                            st.dataframe(scaling_df, use_container_width=True)
                            
                            # Show summary
                            method_counts = scaling_df['Method'].value_counts()
                            st.info(f"📈 Scaling Summary: {', '.join([f'{method}: {count}' for method, count in method_counts.items()])}")
            
    with tab4:
        st.header("🤖 AI-Powered Insights")
        
        if st.session_state.analysis_results is None:
            st.info("ℹ️ No analysis results available. Please run an analysis first.")
        else:
            # LLM-based Recommendations section
            if 'recommendations' in st.session_state.analysis_results:
                recommendations = st.session_state.analysis_results['recommendations']
                
                if 'llm_insights' in recommendations:
                    llm_insights = recommendations['llm_insights']
                    
                    if 'error' in llm_insights:
                        st.error(f"❌ {llm_insights['error']}")
                        if 'fallback_analysis' in llm_insights:
                            st.write("**Fallback Analysis:**")
                            fallback = llm_insights['fallback_analysis']
                            if 'performance_analysis' in fallback:
                                st.write(f"**Performance Analysis:** {fallback['performance_analysis']}")
                            if 'recommendations' in fallback:
                                st.write("**Recommendations:**")
                                for rec in fallback['recommendations']:
                                    st.write(f"• {rec}")
                            if 'summary' in fallback:
                                st.write(f"**Summary:** {fallback['summary']}")
                    else:
                        # Display recommendations only
                        st.markdown("### 💡 AI Recommendations")
                        
                        # Recommendations
                        if 'recommendations' in llm_insights:
                            recs = llm_insights['recommendations']
                            
                            if 'model_improvements' in recs and recs['model_improvements']:
                                st.subheader("🚀 Model Improvements")
                                for i, improvement in enumerate(recs['model_improvements'], 1):
                                    st.markdown(f"**{i}.** {improvement}")
                                    st.markdown("---")
                            
                            if 'feature_engineering_enhancements' in recs and recs['feature_engineering_enhancements']:
                                st.subheader("🔧 Feature Engineering Enhancements")
                                for i, enhancement in enumerate(recs['feature_engineering_enhancements'], 1):
                                    st.markdown(f"**{i}.** {enhancement}")
                                    st.markdown("---")
                            
                            if 'data_collection_suggestions' in recs and recs['data_collection_suggestions']:
                                st.subheader("📥 Data Collection Suggestions")
                                for i, suggestion in enumerate(recs['data_collection_suggestions'], 1):
                                    st.markdown(f"**{i}.** {suggestion}")
                                    st.markdown("---")
                
                elif 'error' in recommendations:
                    st.error(f"❌ {recommendations['error']}")
                else:
                    st.info("ℹ️ No LLM insights available")

    # Footer with enhanced styling
    st.markdown('<div class="footer">Built using Gemini 2.5 Flash and Streamlit | © SubhamModa 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
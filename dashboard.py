"""
Ultra-efficient Streamlit dashboard with real-time predictions.
Optimized for user experience and performance.
"""
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="🚀 Customer Churn Prediction by Hasibur",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = "http://localhost:8000"

class ChurnDashboard:
    """High-performance churn prediction dashboard."""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.timeout = 10
    
    def check_api_health(self) -> bool:
        """Check if API is available."""
        try:
            response = self.session.get(f"{self.api_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def predict_customer(self, customer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call prediction API with error handling."""
        try:
            response = self.session.post(
                f"{self.api_url}/predict",
                json=customer_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API Error: {e}")
            return None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.api_url}/model/info")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def create_customer_form(self) -> Dict[str, Any]:
        """Create optimized customer input form."""
        st.sidebar.header("📋 Customer Information")
        
        # Basic information
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID", value="CUST001", key="customer_id")
            tenure = st.slider("Tenure (months)", 0, 100, 24, key="tenure")
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 
                min_value=0.0, 
                max_value=200.0, 
                value=70.0, 
                step=0.01,
                key="monthly_charges"
            )
        
        with col2:
            total_charges = st.number_input(
                "Total Charges ($)", 
                min_value=0.0, 
                value=1680.0, 
                step=0.01,
                key="total_charges"
            )
            contract = st.selectbox(
                "Contract", 
                ["Month-to-month", "One year", "Two year"],
                key="contract"
            )
            payment_method = st.selectbox(
                "Payment Method", 
                [
                    "Electronic check", 
                    "Mailed check", 
                    "Bank transfer (automatic)", 
                    "Credit card (automatic)"
                ],
                key="payment_method"
            )
        
        # Services
        st.sidebar.subheader("🔧 Services")
        internet_service = st.selectbox(
            "Internet Service", 
            ["DSL", "Fiber optic", "No"],
            key="internet_service"
        )
        phone_service = st.selectbox(
            "Phone Service", 
            ["Yes", "No"],
            key="phone_service"
        )
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            online_security = st.selectbox(
                "Online Security", 
                ["No", "Yes", "No internet service"],
                key="online_security"
            )
            tech_support = st.selectbox(
                "Tech Support", 
                ["No", "Yes", "No internet service"],
                key="tech_support"
            )
        
        with col4:
            streaming_tv = st.selectbox(
                "Streaming TV", 
                ["No", "Yes", "No internet service"],
                key="streaming_tv"
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", 
                ["No", "Yes", "No internet service"],
                key="streaming_movies"
            )
        
        return {
            "customerID": customer_id,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "PhoneService": phone_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "PaymentMethod": payment_method
        }
    
    def create_risk_gauge(self, churn_probability: float) -> go.Figure:
        """Create risk gauge visualization."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk %"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_feature_importance_chart(self, critical_factors: list) -> go.Figure:
        """Create feature importance visualization."""
        if not critical_factors:
            return go.Figure()
        
        df_factors = pd.DataFrame(critical_factors)
        df_factors = df_factors.sort_values('importance', ascending=True)
        
        fig = px.bar(
            df_factors,
            x='impact',
            y='feature',
            orientation='h',
            title="Critical Risk Factors",
            color='impact',
            color_continuous_scale='RdBu_r',
            labels={'impact': 'Impact on Churn Risk', 'feature': 'Feature'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def display_prediction_results(self, result: Dict[str, Any]):
        """Display prediction results with visualizations."""
        if not result:
            st.error("❌ Failed to get prediction results")
            return
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Churn Probability",
                f"{result['churn_probability']:.1%}",
                delta=None
            )
        
        with col2:
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.metric(
                "Risk Level",
                f"{risk_color.get(result['churn_risk'], '⚪')} {result['churn_risk']}"
            )
        
        with col3:
            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}"
            )
        
        with col4:
            st.metric(
                "Processing Time",
                f"{result['processing_time_ms']:.1f}ms"
            )
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("📊 Risk Analysis")
            gauge_fig = self.create_risk_gauge(result['churn_probability'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col_viz2:
            st.subheader("🎯 Critical Factors")
            if 'critical_factors' in result and result['critical_factors']:
                factors_fig = self.create_feature_importance_chart(result['critical_factors'])
                st.plotly_chart(factors_fig, use_container_width=True)
            else:
                st.info("Feature importance analysis available in advanced mode")
        
        # Business insights
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            st.subheader("💡 Business Explanation")
            if 'explanation' in result:
                st.info(result['explanation'])
            else:
                st.info(f"Customer {result['customer_id']} has a {result['churn_risk']} risk of churning with {result['churn_probability']:.1%} probability.")
            
            st.subheader("📧 Personalized Email")
            email_text = result.get('personalized_email', f"Dear {result['customer_id']},\n\nBased on our analysis, we'd like to offer you personalized retention options to improve your experience with us.")
            st.text_area(
                "Retention Email",
                value=email_text,
                height=150,
                disabled=True
            )
        
        with col_insights2:
            st.subheader("🎯 Retention Strategy")
            strategy = result.get('retention_strategy', f"Based on the {result['churn_risk']} risk level, we recommend proactive engagement and personalized offers.")
            st.success(strategy)
            
            # Additional insights
            st.subheader("📈 Performance Metrics")
            st.json({
                "Processing Time": f"{result['processing_time_ms']:.1f}ms",
                "API Response": "Ultra-fast",
                "Model Accuracy": "Optimized"
            })
    
    def display_model_info(self):
        """Display model information."""
        model_info = self.get_model_info()
        if model_info:
            st.subheader("🤖 Model Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Type", model_info.get('model_type', 'Unknown'))
                st.metric("Feature Count", model_info.get('feature_count', 0))
            
            with col2:
                cv_scores = model_info.get('cv_scores', {})
                if cv_scores:
                    st.metric("CV AUC Score", f"{cv_scores.get('mean', 0):.3f}")
                    st.metric("CV Std Dev", f"{cv_scores.get('std', 0):.3f}")
            
            if model_info.get('top_features'):
                st.subheader("🔝 Top Features")
                st.write(", ".join(map(str, model_info['top_features'][:10])))
    
    def run(self):
        """Run the dashboard."""
        # Header
        st.title("🚀 Customer Churn Prediction Dashboard")
        st.markdown("Real-time churn prediction with AI-powered business insights by Hasibur")
        
        # Check API health
        if not self.check_api_health():
            st.error("❌ API is not available. Please ensure the API server is running.")
            st.stop()
        
        # Main layout
        col_main1, col_main2 = st.columns([1, 2])
        
        with col_main1:
            # Customer form
            customer_data = self.create_customer_form()
            
            # Predict button
            if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
                with st.spinner("Analyzing customer data..."):
                    result = self.predict_customer(customer_data)
                    if result:
                        st.session_state['prediction_result'] = result
        
            # Model info
            with st.expander("ℹ️ Model Information"):
                self.display_model_info()
        
        with col_main2:
            # Display results
            if 'prediction_result' in st.session_state:
                self.display_prediction_results(st.session_state['prediction_result'])
            else:
                st.info("👈 Enter customer information and click 'Predict Churn Risk' to get started")
                
                # Sample visualization
                st.subheader("📊 Sample Risk Distribution")
                sample_data = pd.DataFrame({
                    'Risk Level': ['Low', 'Medium', 'High'],
                    'Percentage': [45, 35, 20]
                })
                
                fig = px.pie(
                    sample_data, 
                    values='Percentage', 
                    names='Risk Level',
                    title="Typical Customer Risk Distribution",
                    color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the dashboard."""
    dashboard = ChurnDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()

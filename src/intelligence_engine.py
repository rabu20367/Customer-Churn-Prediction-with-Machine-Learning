"""
SHAP + LLM fusion for real-time business intelligence.
Ultra-fast insights generation with minimal API calls.
Developed by Hasibur.
"""
try:
    import shap
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Import error in intelligence_engine: {e}")
    # These imports are handled at runtime
from typing import Dict, Any, List, Optional
import logging
import os
from config import Config

logger = logging.getLogger(__name__)

class IntelligenceEngine:
    """
    High-performance intelligence engine combining SHAP and LLM.
    Optimized for real-time business insights generation.
    """
    
    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None
        self.groq_client: Optional[Any] = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize SHAP explainer and LLM client."""
        try:
            # Initialize SHAP explainer
            if hasattr(self.model, 'base_estimator'):
                base_model = self.model.base_estimator
            else:
                base_model = self.model
            
            self.explainer = shap.TreeExplainer(base_model)
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
        
        # Initialize Groq client
        try:
            if Config.GROQ_API_KEY:
                import groq
                self.groq_client = groq.Groq(api_key=Config.GROQ_API_KEY)
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("Groq API key not configured")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
    
    def extract_critical_factors(self, shap_values: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Extract top contributing factors from SHAP values.
        Optimized for business interpretation.
        """
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Get top contributing features
        feature_importance = np.abs(shap_values)
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        
        critical_factors = []
        for idx in top_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            critical_factors.append({
                'feature': feature_name,
                'impact': float(shap_values[idx]),
                'importance': float(feature_importance[idx]),
                'direction': 'increases' if shap_values[idx] > 0 else 'decreases'
            })
        
        return critical_factors
    
    def generate_business_insight(self, customer_data: Dict[str, Any], 
                                prediction: float, 
                                critical_factors: List[Dict[str, Any]]) -> str:
        """
        Generate concise business explanation using LLM.
        Optimized for single API call efficiency.
        """
        if not self.groq_client:
            return self._generate_fallback_explanation(prediction, critical_factors)
        
        # Prepare context
        context = self._prepare_business_context(customer_data, prediction, critical_factors)
        
        prompt = f"""
        Customer churn analysis for business stakeholders:
        
        {context}
        
        Generate a concise business explanation (2-3 sentences) that:
        1. States the churn risk level clearly
        2. Identifies the primary risk factor
        3. Suggests immediate action
        
        Focus on business impact, not technical details. Be specific and actionable.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_fallback_explanation(prediction, critical_factors)
    
    def generate_retention_strategy(self, customer_data: Dict[str, Any], 
                                  critical_factors: List[Dict[str, Any]]) -> str:
        """
        Generate specific retention strategy.
        """
        if not self.groq_client:
            return self._generate_fallback_strategy(critical_factors)
        
        prompt = f"""
        Generate ONE specific retention strategy for this customer:
        
        Customer: {customer_data.get('customerID', 'Customer')}
        Tenure: {customer_data.get('tenure', 'N/A')} months
        Contract: {customer_data.get('Contract', 'N/A')}
        Risk factors: {', '.join([f['feature'] for f in critical_factors[:3]])}
        
        Provide ONE actionable strategy (1-2 sentences) that addresses the main risk factor.
        Be specific and cost-effective.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_fallback_strategy(critical_factors)
    
    def generate_personalized_email(self, customer_data: Dict[str, Any], 
                                  prediction: float, 
                                  critical_factors: List[Dict[str, Any]]) -> str:
        """
        Generate personalized retention email opener.
        """
        if not self.groq_client:
            return self._generate_fallback_email(customer_data)
        
        prompt = f"""
        Write a personalized email opener for this customer:
        
        Customer: {customer_data.get('customerID', 'Valued Customer')}
        Tenure: {customer_data.get('tenure', 'N/A')} months
        Risk Level: {prediction:.1%}
        Main Issue: {critical_factors[0]['feature'] if critical_factors else 'General concern'}
        
        Requirements:
        - Warm, personal tone
        - Acknowledge their value
        - Address their specific situation
        - Under 50 words
        
        Make it feel personal, not templated.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_fallback_email(customer_data)
    
    def generate_complete_insight(self, customer_data: Dict[str, Any], 
                                prediction: float) -> Dict[str, str]:
        """
        Generate complete business insight in optimized single flow.
        Minimizes API calls while maximizing information value.
        """
        # Get SHAP values
        customer_df = pd.DataFrame([customer_data])
        X_processed = self.scaler.transform(customer_df)
        
        critical_factors = []
        if self.explainer:
            try:
                shap_values = self.explainer.shap_values(X_processed)[1]  # Positive class
                critical_factors = self.extract_critical_factors(shap_values)
            except Exception as e:
                logger.error(f"SHAP calculation error: {e}")
        
        # Generate insights
        explanation = self.generate_business_insight(customer_data, prediction, critical_factors)
        strategy = self.generate_retention_strategy(customer_data, critical_factors)
        email = self.generate_personalized_email(customer_data, prediction, critical_factors)
        
        return {
            'explanation': explanation,
            'retention_strategy': strategy,
            'personalized_email': email,
            'critical_factors': critical_factors
        }
    
    def _prepare_business_context(self, customer_data: Dict[str, Any], 
                                prediction: float, 
                                critical_factors: List[Dict[str, Any]]) -> str:
        """Prepare business context for LLM prompts."""
        context_parts = [
            f"Customer Profile:",
            f"- Tenure: {customer_data.get('tenure', 'N/A')} months",
            f"- Monthly Charges: ${customer_data.get('MonthlyCharges', 'N/A')}",
            f"- Contract: {customer_data.get('Contract', 'N/A')}",
            f"- Payment Method: {customer_data.get('PaymentMethod', 'N/A')}",
            f"",
            f"Churn Probability: {prediction:.1%}",
            f"",
            f"Key Risk Factors:"
        ]
        
        for factor in critical_factors[:3]:
            context_parts.append(f"- {factor['feature']}: {factor['direction']} churn risk")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_explanation(self, prediction: float, 
                                     critical_factors: List[Dict[str, Any]]) -> str:
        """Fallback explanation when LLM is unavailable."""
        risk_level = "High" if prediction > 0.7 else "Medium" if prediction > 0.4 else "Low"
        main_factor = critical_factors[0]['feature'] if critical_factors else "general factors"
        
        return f"Customer has {risk_level.lower()} churn risk ({prediction:.1%}). Primary risk factor: {main_factor}. Immediate action recommended."
    
    def _generate_fallback_strategy(self, critical_factors: List[Dict[str, Any]]) -> str:
        """Fallback strategy when LLM is unavailable."""
        if not critical_factors:
            return "Contact customer to understand their needs and offer personalized solutions."
        
        main_factor = critical_factors[0]['feature']
        if 'Contract' in main_factor:
            return "Offer contract extension incentives or flexible payment options."
        elif 'tenure' in main_factor.lower():
            return "Provide loyalty rewards and personalized service to increase engagement."
        else:
            return "Address specific service concerns and offer premium support."
    
    def _generate_fallback_email(self, customer_data: Dict[str, Any]) -> str:
        """Fallback email when LLM is unavailable."""
        customer_id = customer_data.get('customerID', 'Valued Customer')
        return f"Dear {customer_id}, we value your business and want to ensure you're getting the most from our services. Let's discuss how we can better serve you."

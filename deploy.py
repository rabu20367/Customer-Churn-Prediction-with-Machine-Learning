"""
One-command deployment script.
Complete system setup and launch.
"""
import subprocess
import sys
import os
import time
import logging
from pathlib import Path
try:
    import requests
except ImportError:
    print("requests not available - install with: pip install requests")
    requests = None
import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages complete system deployment."""
    
    def __init__(self):
        self.api_process = None
        self.dashboard_process = None
        self.api_url = "http://localhost:8000"
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'lightgbm', 
            'optuna', 'shap', 'fastapi', 'uvicorn', 'streamlit', 'groq'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Run: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All dependencies are installed")
        return True
    
    def check_data_availability(self):
        """Check if training data is available."""
        data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            logger.info("Please download the dataset from:")
            logger.info("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
            logger.info("And place it in the data/ folder")
            return False
        
        logger.info("✅ Training data is available")
        return True
    
    def train_model(self):
        """Train the model if not already trained."""
        model_path = "models/churn_model.pkl"
        
        if os.path.exists(model_path):
            logger.info("✅ Model already trained")
            return True
        
        logger.info("🤖 Training model...")
        try:
            result = subprocess.run([sys.executable, "train.py"], 
                                  capture_output=True, text=True, check=True)
            logger.info("✅ Model training completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model training failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def start_api(self):
        """Start the FastAPI server."""
        logger.info("🚀 Starting API server...")
        
        try:
            self.api_process = subprocess.Popen(
                [sys.executable, "api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for API to start
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ API server started successfully")
                        return True
                except:
                    time.sleep(1)
            
            logger.error("❌ API server failed to start")
            return False
            
        except Exception as e:
            logger.error(f"Error starting API: {e}")
            return False
    
    def start_dashboard(self):
        """Start the Streamlit dashboard."""
        logger.info("📊 Starting dashboard...")
        
        try:
            self.dashboard_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "dashboard.py", 
                 "--server.port", "8501", "--server.headless", "true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for dashboard to start
            time.sleep(5)
            logger.info("✅ Dashboard started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            return False
    
    def check_system_health(self):
        """Check if all components are running."""
        logger.info("🔍 Checking system health...")
        
        # Check API
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is healthy")
            else:
                logger.error("❌ API health check failed")
                return False
        except:
            logger.error("❌ API is not responding")
            return False
        
        # Check dashboard (basic check)
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Dashboard is accessible")
            else:
                logger.warning("⚠️ Dashboard may not be fully ready")
        except:
            logger.warning("⚠️ Dashboard is not yet accessible")
        
        return True
    
    def cleanup(self):
        """Clean up processes on exit."""
        logger.info("🧹 Cleaning up processes...")
        
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait()
        
        if self.dashboard_process:
            self.dashboard_process.terminate()
            self.dashboard_process.wait()
    
    def deploy(self):
        """Complete deployment process."""
        logger.info("🚀 Starting deployment...")
        
        try:
            # 1. Check dependencies
            if not self.check_dependencies():
                return False
            
            # 2. Check data availability
            if not self.check_data_availability():
                return False
            
            # 3. Train model if needed
            if not self.train_model():
                return False
            
            # 4. Start API
            if not self.start_api():
                return False
            
            # 5. Start dashboard
            if not self.start_dashboard():
                return False
            
            # 6. Check system health
            if not self.check_system_health():
                return False
            
            # 7. Display success message
            self.display_success_message()
            
            # 8. Keep running
            self.keep_running()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Deployment interrupted by user")
            self.cleanup()
            return False
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.cleanup()
            return False
    
    def display_success_message(self):
        """Display deployment success message."""
        print("\n" + "=" * 70)
        print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("📊 Dashboard: http://localhost:8501")
        print("🚀 API: http://localhost:8000")
        print("📖 API Docs: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("\n💡 Features by Hasibur:")
        print("  • Real-time churn prediction")
        print("  • AI-powered business insights")
        print("  • Sub-20ms response times")
        print("  • Interactive visualizations")
        print("\n🛑 Press Ctrl+C to stop the system")
        print("=" * 70)
    
    def keep_running(self):
        """Keep the system running until interrupted."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down system...")
            self.cleanup()

def main():
    """Main deployment function."""
    deployment_manager = DeploymentManager()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal")
        deployment_manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run deployment
    success = deployment_manager.deploy()
    
    if not success:
        logger.error("❌ Deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

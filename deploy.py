"""
One-command deployment script.
Complete system setup and launch.
"""
import subprocess
import sys
import os
import time
import logging
import json
import traceback
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
        """Clean up processes and artifacts on exit."""
        logger.info("🧹 Cleaning up processes and artifacts...")
        
        # Terminate processes
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=5)
                logger.info("✅ API process terminated")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ API process did not terminate gracefully, forcing kill")
                self.api_process.kill()
            except Exception as e:
                logger.error(f"Error terminating API process: {e}")
        
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                logger.info("✅ Dashboard process terminated")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Dashboard process did not terminate gracefully, forcing kill")
                self.dashboard_process.kill()
            except Exception as e:
                logger.error(f"Error terminating dashboard process: {e}")
        
        # Clean up partial artifacts if deployment failed
        self._cleanup_partial_artifacts()
    
    def _cleanup_partial_artifacts(self):
        """Clean up partial artifacts from failed deployment."""
        try:
            # Check for partial model files
            model_path = "models/churn_model.pkl"
            if os.path.exists(model_path) and os.path.getsize(model_path) < 1000:  # Suspiciously small
                logger.warning("⚠️ Removing partial model file")
                os.remove(model_path)
            
            # Check for partial data engine files
            data_engine_path = "models/data_engine.pkl"
            if os.path.exists(data_engine_path) and os.path.getsize(data_engine_path) < 100:  # Suspiciously small
                logger.warning("⚠️ Removing partial data engine file")
                os.remove(data_engine_path)
                
        except Exception as e:
            logger.error(f"Error during artifact cleanup: {e}")
    
    def _log_deployment_error(self, error: Exception, context: str):
        """Log deployment errors with context."""
        error_log = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        # Log to file
        try:
            with open("deployment_error.log", "a") as f:
                f.write(f"\n{'='*50}\n")
                f.write(json.dumps(error_log, indent=2))
                f.write(f"\n{'='*50}\n")
        except Exception as log_error:
            logger.error(f"Failed to write error log: {log_error}")
        
        logger.error(f"Deployment error in {context}: {error}")
    
    def deploy(self):
        """Complete deployment process with comprehensive error handling."""
        logger.info("🚀 Starting deployment...")
        
        try:
            # 1. Check dependencies
            logger.info("Step 1/6: Checking dependencies...")
            if not self.check_dependencies():
                self._log_deployment_error(Exception("Dependency check failed"), "dependency_check")
                return False
            
            # 2. Check data availability
            logger.info("Step 2/6: Checking data availability...")
            if not self.check_data_availability():
                self._log_deployment_error(Exception("Data not available"), "data_check")
                return False
            
            # 3. Train model if needed
            logger.info("Step 3/6: Training model...")
            if not self.train_model():
                self._log_deployment_error(Exception("Model training failed"), "model_training")
                return False
            
            # 4. Start API
            logger.info("Step 4/6: Starting API server...")
            if not self.start_api():
                self._log_deployment_error(Exception("API startup failed"), "api_startup")
                return False
            
            # 5. Start dashboard
            logger.info("Step 5/6: Starting dashboard...")
            if not self.start_dashboard():
                self._log_deployment_error(Exception("Dashboard startup failed"), "dashboard_startup")
                return False
            
            # 6. Check system health
            logger.info("Step 6/6: Verifying system health...")
            if not self.check_system_health():
                self._log_deployment_error(Exception("System health check failed"), "health_check")
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
            self._log_deployment_error(e, "deployment_main")
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

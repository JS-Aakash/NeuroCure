
import subprocess
import sys
import time
import os
import threading
from pathlib import Path

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("Backend stopped")
    except Exception as e:
        print(f"Backend error: {e}")

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    time.sleep(5)
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("Frontend stopped")
    except Exception as e:
        print(f"Frontend error: {e}")

def check_files():
    """Check if required files exist"""
    required_files = ['app.py', 'frontend.py']
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    
    if not Path('.env').exists():
        print("⚠️  No .env file found. Creating one...")
        with open('.env', 'w') as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("\n# Optional: Other API keys\n")
            f.write("# FDA_API_KEY=your_fda_api_key_here\n")
        print("✅ .env file created. Please add your OpenAI API key.")
        return False
    
    return True

def main():
    print("🏥 AI Medical Prescription Verification System")
    print("=" * 50)
    
    if not check_files():
        input("Press Enter to exit...")
        return
    
    print("✅ All required files found")
    
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    
    try:
        print("\n🌐 Starting services...")
        print("📡 Backend API will be at: http://localhost:8000")
        print("🖥️  Frontend UI will be at: http://localhost:8501")
        print("\n⚠️  Press Ctrl+C to stop both services")
        
        backend_thread.start()
        frontend_thread.start()
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
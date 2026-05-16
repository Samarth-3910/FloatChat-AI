import subprocess
import sys
import os
import time
import requests

REQUIRED_FILES = ["app.py", "oceanography_dashboard.py", "requirements.txt"]
BACKEND_URL = "http://127.0.0.1:5000"
FRONTEND_URL = "http://localhost:8501"

def check_requirements():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    print("✅ All required files found!")
    return True
 
def install_requirements():
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def wait_for_backend(timeout=30):
    """Wait until Flask backend is reachable."""
    print(f"⏳ Waiting for backend ({BACKEND_URL}) to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(BACKEND_URL)
            if r.status_code < 500:
                print("✅ Backend is ready!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print(f"❌ Backend did not respond within {timeout} seconds.")
    return False

def run_backend():
    print("🚀 Starting backend...")
    return subprocess.Popen([sys.executable, "app.py"])
 
def run_frontend():
    print("🌐 Starting frontend (Streamlit)...")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "oceanography_dashboard.py"], shell=True)

def main():
    print("="*40)
    print("📦 RAG Assistant Auto-Launcher")
    print("="*40)

    if not check_requirements():
        return

    user_input = input("\n📥 Install requirements before launching? (y/n): ").strip().lower()
    if user_input in ("y", "yes"):
        if not install_requirements():
            return

    # Start backend
    backend_proc = run_backend()
    if not wait_for_backend(timeout=30):
        print("🛑 Exiting due to backend startup failure.")
        backend_proc.terminate()
        return

    # Start frontend
    frontend_proc = run_frontend()

    print(f"\n✅ Application launched successfully!")
    print(f"🔗 Frontend: {FRONTEND_URL}")
    print(f"🩺 Backend: {BACKEND_URL}")
    print("\n⏹️ Press Ctrl+C to stop both processes.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down processes...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("👋 Exited cleanly.")

if __name__ == "__main__":
    main()

"""setup.py — Run this first to set up the project environment."""
import subprocess, sys, os

def main():
    print("Setting up Mental Health Chatbot...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
    os.chdir("frontend")
    subprocess.check_call(["npm", "install"])
    os.chdir("..")
    print("Done! Copy backend/.env.example to backend/.env and fill in your keys.")

if __name__ == "__main__":
    main()

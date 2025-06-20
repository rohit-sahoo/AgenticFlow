"""
Launcher script for IntellectDesign v2 with LangGraph orchestration.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.main import main

if __name__ == "__main__":
    print("Starting IntellectDesign with LangGraph orchestration...")
    main() 
#!/usr/bin/env python3
"""
ML Pipeline Runner
Runs all notebooks in the correct order
"""

import subprocess
import sys
from pathlib import Path

# Define the pipeline order
PIPELINE = [
    {
        "name": "Feature Engineering",
        "file": "cervical_02_feature_engineering_advanced.ipynb",
        "description": "Creates processed_data/*.npy files from raw data"
    },
    {
        "name": "Model Training with Hyperparameter Tuning",
        "file": "cervical_03_model_training_advanced.ipynb",
        "description": "Trains regularized models with leakage-safe repeated CV"
    },
    {
        "name": "MLflow Model Registry",
        "file": "cervical_04_mlflow_model_registry.ipynb",
        "description": "Registers models in MLflow"
    },
    {
        "name": "Model Evaluation",
        "file": "cervical_04_model_evaluation.ipynb",
        "description": "Evaluates all models and writes evaluation_outputs/"
    }
]

def run_notebook(notebook_path):
    """Run a Jupyter notebook using nbconvert"""
    print(f"\n{'='*70}")
    print(f"Running: {notebook_path}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "-m", "jupyter",
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {notebook_path}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False

def main():
    base_dir = Path(__file__).parent
    notebooks_dir = base_dir / "notebooks"
    
    print("="*70)
    print("ML PIPELINE RUNNER")
    print("="*70)
    print("\nThis script will run all notebooks in the correct order:")
    for i, step in enumerate(PIPELINE, 1):
        print(f"\n{i}. {step['name']}")
        print(f"   File: {step['file']}")
        print(f"   Purpose: {step['description']}")
    
    print("\n" + "="*70)
    response = input("\nDo you want to proceed? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Pipeline cancelled.")
        return
    
    # Run each notebook in order
    for step in PIPELINE:
        notebook_path = notebooks_dir / step['file']
        
        if not notebook_path.exists():
            print(f"❌ Notebook not found: {notebook_path}")
            sys.exit(1)
        
        success = run_notebook(notebook_path)
        
        if not success:
            print(f"\n❌ Pipeline failed at: {step['name']}")
            print("Please fix the error and run again.")
            sys.exit(1)
        
        print(f"✅ {step['name']} completed successfully")
    
    print("\n" + "="*70)
    print("✅ ENTIRE PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("  1. View MLflow UI: mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db")
    print("  2. Check models in: models/")
    print("  3. Start API: cd api && python run.py")

if __name__ == "__main__":
    main()
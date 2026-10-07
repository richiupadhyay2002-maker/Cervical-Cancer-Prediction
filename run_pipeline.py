#!/usr/bin/env python
"""
Run the Cervical Cancer Prediction pipeline end to end.

Executes the notebooks in order with `jupyter nbconvert --execute --inplace`:

  1. Feature engineering  -> processed_data/*.npy, processed_data/feature_columns.json
  2. Model training       -> models/*_tuned.pkl, models/model_comparison_advanced.csv,
                             runs logged to notebooks/mlflow.db
  3. Model evaluation     -> evaluation_outputs/*
  4. MLflow registry      -> registers all models in notebooks/mlflow.db and promotes
                             the model selected by validation F1 to Production

EDA (notebooks/cervical_eda.ipynb) is exploratory and is not part of the pipeline.

Usage:
    python run_pipeline.py          # asks for confirmation
    python run_pipeline.py --yes    # non-interactive
"""
import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE = [
    ("Feature Engineering", "cervical_02_feature_engineering_advanced.ipynb"),
    ("Model Training with Hyperparameter Tuning", "cervical_03_model_training_advanced.ipynb"),
    ("Model Evaluation", "cervical_04_model_evaluation.ipynb"),
    ("MLflow Model Registry", "cervical_04_mlflow_model_registry.ipynb"),
]


def run_notebook(notebook_path: Path) -> bool:
    print(f"\n{'=' * 70}\nRunning: {notebook_path.name}\n{'=' * 70}")
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=1800",
            str(notebook_path),
        ],
        cwd=notebook_path.parent,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="run without the confirmation prompt")
    args = parser.parse_args()

    notebooks_dir = Path(__file__).resolve().parent / "notebooks"
    print("Pipeline steps:")
    for i, (name, file) in enumerate(PIPELINE, 1):
        print(f"  {i}. {name} ({file})")
    print("\nNote: training overwrites the tracked models/*.pkl and evaluation_outputs/* files.")

    if not args.yes and input("\nProceed? (yes/no): ").strip().lower() not in {"y", "yes"}:
        print("Cancelled.")
        return

    for name, file in PIPELINE:
        path = notebooks_dir / file
        if not path.exists():
            sys.exit(f"Notebook not found: {path}")
        if not run_notebook(path):
            sys.exit(f"\nPipeline failed at: {name}")
        print(f"Done: {name}")

    print("\nPipeline completed.")
    print("Next steps:")
    print("  1. MLflow UI:  cd notebooks && mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  2. Start API:  cd api && python run.py   (Swagger UI at http://localhost:8000/docs)")


if __name__ == "__main__":
    main()

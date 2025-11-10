# DSCommittee

Lightweight end-to-end pipeline for text classification / regression.  
Provides preprocessing, model training, saving/loading, evaluation, simple stacking and CSV submission export.

## Table of contents
- [Features](#features)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Model saving / loading conventions](#model-saving--loading-conventions)
- [Notes & known issues](#notes--known-issues)
- [Dependencies](#dependencies)
- [How to contribute](#how-to-contribute)

## Features
- Text cleaning and TF-IDF vectorization.
- Label encoding and class filtering.
- Training helper functions for sklearn, CatBoost and PyTorch (skorch).
- Save / load trained models for later evaluation.
- Stacking: sklearn stack, PyTorch meta-learner, combined meta-learner.
- Export predictions to CSV for submission.

## Repository structure
- Main.ipynb — primary notebook: data load → preprocess → train/evaluate → stack → export.
- Data_Editing_Helpers.py — data helpers (wrangling, caching, save/load helpers).
- Classifier.py — classifier training functions (sklearn, CatBoost, XGBoost, Skorch/PyTorch).
- Regressor.py — regressor training functions.
- converter.ipynb — utilities to convert raw .txt files to .csv.
- TrainedModels/ — trained model artifacts (see conventions).
- Data/ — input CSVs (train/test).
- README.md — this file.

## Quick start
1. Place CSVs in `Data/`. Use `converter.ipynb` to convert raw .txt datasets if needed.
2. Open `Main.ipynb` and set:
   - `y_name` (target column)
   - `x_name` (id/drop column)
   - `is_regression` flag
3. Run preprocessing cells to create `vectorizer`, `X_train`, `X_test`, `y_train`, `y_test`.
4. To evaluate only (no training): run the "Loading Models" cell — it loads models from `TrainedModels/` and evaluates on test data.
5. To (re)train models: run training cells in `Main.ipynb` (these call functions in `Classifier.py` / `Regressor.py` and save models).
6. Stacking and submission export are implemented in subsequent cells and save CSV files in the repo root.

## Model saving / loading conventions
- Sklearn models & Pipelines: save with `joblib.dump(model, './TrainedModels/<name>.pkl')` and load with `joblib.load`.
- PyTorch bundles: save a dict with keys: `model_state_dict`, `full_model` (optional), `vocab` using `torch.save(..., '<name>.pth')`.
- Loading cell supports both `.pkl` (joblib / pickle) and `.pth` bundles (uses `torch.load(..., map_location='cpu')`).

## Notes & known issues
- Use `joblib.load` for sklearn pipelines to avoid unpickling errors. Import local modules (e.g. `Classifier`, `Regressor`, `Data_Editing_Helpers`) in the running kernel before loading models to ensure referenced classes are available.
- PyTorch bundles require the same model class/architecture to be importable to load a full model or to reconstitute from `model_state_dict`.
- If you see errors like `UnpicklingError: STACK_GLOBAL requires str`, models were likely pickled in an incompatible environment — retraining in the current environment may be required.
- Ensure consistent Python / package versions when sharing saved models.

## Dependencies (high level)
- Python 3.8+
- pandas, numpy, scikit-learn, joblib
- torch, skorch, catboost, xgboost (optional, only if used)
- nltk (stopwords)
- Install via pip all packages:  
  pip install -r requirements.txt

## How to contribute
- Open an issue or submit a PR.
- Keep model save/load behavior consistent (prefer joblib for sklearn).
- Add unit tests for helper modules and training functions.

## License
- Project for experimentation. No explicit license file included — add one if you intend to publish.

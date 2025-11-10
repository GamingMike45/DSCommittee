import Data_Editing_Helpers as DEH
import xgboost as xgb
import torch
import numpy as np
import torch.nn.functional as F
import requests
import re
import zipfile
import os
import pandas as pd
import dill
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from torch import nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from skorch.callbacks import GradientNormClipping
from skorch.callbacks import LRScheduler


def random_search_cv(model, param_grid, X_train, y_train, n_iter, cv=None, n_jobs=None):
    """Run RandomizedSearchCV with full parallelism"""
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        cv=cv if cv is not None else 5,
        scoring="accuracy",
        n_iter=n_iter,
        random_state=1103,
        n_jobs=n_jobs if n_jobs is not None else 4,  # default 4 if not set
        verbose=2
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# ---------------- CPU Tree-Based Models ----------------
def decisiontreeClassifier(X_train, y_train):
    print("\nDecision Tree Classifier Started")
    model = DecisionTreeClassifier(random_state=301)
    param_grid = {
        "max_depth": [1, 3, 5, 7, 10],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_leaf_nodes": [None, 5, 10, 15],
        "min_weight_fraction_leaf": [0.0, 0.1, 0.2]
    }
    best_model, best_params = random_search_cv(model, param_grid, X_train, y_train, n_iter=20)
    print(f"Best parameters Decision Tree: {best_params}")
    DEH.saveModel(best_model, "./TrainedModels/decisiontreeClassifier.pkl")
    print("Decision Tree Classifier Finished")
    return best_model


def extraTreesClassifier(X_train, y_train):
    print("\nExtraTrees Classifier Started")
    model = ExtraTreesClassifier(random_state=301)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10]
    }
    best_model, best_params = random_search_cv(model, param_grid, X_train, y_train, n_iter=15)
    print(f"Best parameters ExtraTrees: {best_params}")
    DEH.saveModel(best_model, "./TrainedModels/extraTreesClassifier.pkl")
    print("ExtraTrees Classifier Finished")
    return best_model


def adaboostClassifier(X_train, y_train):
    print("\nAdaBoost Classifier Started")
    model = AdaBoostClassifier()
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    }
    search = GridSearchCV(model, param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    best_model, best_params = search.best_estimator_, search.best_params_
    print(f"Best parameters AdaBoost: {best_params}")
    DEH.saveModel(best_model, "./TrainedModels/adaBoostClassifier.pkl")
    print("AdaBoost Classifier Finished")
    return best_model


# ---------------- GPU / Fast Tree-Based Models ----------------
def catBoostClassifier(X_train, y_train):
    import catboost
    print("\nCatBoost Classifier Started (GPU Enabled)")

    # Force GPU usage
    model = catboost.CatBoostClassifier(
        task_type="GPU",      
        devices="0",          
        verbose=100,          
        random_seed=301,
        border_count=128,          
        max_ctr_complexity=2,
        gpu_ram_part=0.95      
    )

    # Confirm GPU availability inside CatBoost
    try:
        print(f"CatBoost is using device: {model.get_params()['task_type']}")
    except Exception as e:
        print(f"Could not confirm device: {e}")

    param_grid = {
    "iterations": [100, 150, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "depth": [2, 3, 5, 7, 9],
    "l2_leaf_reg": [3, 5, 9],
}

    # Run RandomizedSearchCV
    search = random_search_cv(model, param_grid, X_train, y_train, n_iter=10, cv=2, n_jobs=1)
    best_model, best_params = search

    print(f"Best parameters CatBoost: {best_params}")
    DEH.saveModel(best_model, "./TrainedModels/catBoostClassifier.pkl")
    print("CatBoost Classifier Finished")
    return best_model


def xgBoostClassifier(X_train, y_train):
    print("\nXGBoost Classifier Started (GPU Enabled)")
    y_train_enc = LabelEncoder().fit_transform(y_train)
    model = xgb.XGBClassifier(
        tree_method="auto",
        predictor="gpu_predictor",
        device="cuda",
        eval_metric="mlogloss",
        random_state=1103
    )

    try:
        booster = model.get_booster()
        gpu_used = booster.attributes().get('device', 'CPU')
        print(f"XGBoost is using device: {gpu_used}")
    except Exception:
        # If model hasn't been trained yet, device info isn't available
        print("XGBoost device check will occur after training starts (GPU should be used).")
    
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    best_model, best_params = random_search_cv(model, param_grid, X_train, y_train_enc, n_iter=20)
    print(f"Best parameters XGBoost: {best_params}")
    DEH.saveModel(best_model, "./TrainedModels/xgBoostClassifier.pkl")
    print("XGBoost Classifier Finished")
    return best_model


# ---------------- Lightweight Baselines ----------------
def logisticRegressionClassifier(X_train, y_train):
    print("\nLogistic Regression Classifier Started")
    model = LogisticRegression(max_iter=500, solver='saga', n_jobs=-1)
    model.fit(X_train, y_train)
    DEH.saveModel(model, "./TrainedModels/logisticRegressionClassifier.pkl")
    print("Logistic Regression Classifier Finished")
    return model


def sgdClassifier(X_train, y_train):
    print("\nSGD Classifier Started")
    model = SGDClassifier(max_iter=500, tol=1e-3, n_jobs=-1)
    model.fit(X_train, y_train)
    DEH.saveModel(model, "./TrainedModels/sgdClassifier.pkl")
    print("SGD Classifier Finished")
    return model



# ---------------- PyTorch Neural Network Models ----------------
def build_glove_embeddings(X_text, max_vocab_size=20000, embedding_dim=100):

    # Simple tokenizer
    def tokenize(text):
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    # Tokenize all text
    tokenized = [tokenize(t) for t in X_text]
    counter = Counter([tok for sent in tokenized for tok in sent])

    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    def encode_sentence(tokens):
        return torch.tensor([vocab.get(tok, 1) for tok in tokens], dtype=torch.long)

    encoded = [encode_sentence(toks) for toks in tokenized]
    X_padded = pad_sequence(encoded, batch_first=True, padding_value=0)

    # Download GloVe embeddings if needed
    glove_path = "./glove.6B.100d.txt"
    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings ")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = "./glove.6B.zip"
        with open(zip_path, "wb") as f:
            f.write(requests.get(url).content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)

    # Build embedding matrix
    print("Building embedding matrix...")
    embedding_matrix = torch.zeros(len(vocab), embedding_dim)
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vec

    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = torch.tensor(embeddings_index[word])
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1

    print(f"Embedding matrix built: {embedding_matrix.shape}")
    return X_padded, vocab, embedding_matrix


def torch_save_model(obj, path):
    torch.save(obj, path, pickle_module=dill)

def torch_load_model(path):
    return torch.load(path, pickle_module=dill)


def pytorchClassifier(X_text, y_encoded, do_grid_search=True):
    print("\nPyTorch Simple Neural Network (GloVe) Started")

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_encoded).astype(np.int64)

    # Build embeddings
    X_padded, vocab, embedding_matrix = build_glove_embeddings(X_text)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_padded, y, test_size=0.2, stratify=y, random_state=1103)


    num_classes = len(np.unique(y_train))
    embed_dim = embedding_matrix.shape[1]
    hidden_dim = 256

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False, padding_idx=0
            )
            self.fc = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(hidden_dim, num_classes)
            )

        def forward(self, X):
            emb = self.embedding(X)
            mean_emb = emb.mean(dim=1)
            max_emb = emb.max(dim=1).values
            features = torch.cat([mean_emb, max_emb], dim=1)  # double embedding info
            return self.fc(features)

    grad_clip = GradientNormClipping(1.0)
    lr_scheduler = LRScheduler(
    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
    monitor='valid_loss',      # what to monitor
    patience=2,
    factor=0.5,
    )


    early_stop = EarlyStopping(patience=3, monitor='valid_loss', lower_is_better=True, load_best=True)
    model = NeuralNetClassifier(
        module=SimpleNet,
        max_epochs=10,

        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        criterion=nn.CrossEntropyLoss,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=3,
        train_split=lambda ds, y: (
            torch.utils.data.TensorDataset(X_train, torch.tensor(y_train)),
            torch.utils.data.TensorDataset(X_valid, torch.tensor(y_valid))
        ),
        callbacks=[early_stop, grad_clip, lr_scheduler],
    )
    
    X_train_tensor = X_train
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)


    if do_grid_search:
        param_grid = {
            "lr": [0.0005, 0.0003, 0.0001],
            "batch_size": [256, 512, 1024]
        }
        gs = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=1, refit=True, verbose=3)
        gs.fit(X_train_tensor, y_train_tensor)
        model = gs.best_estimator_
        print("Best parameters:", gs.best_params_)
    else:
        model.fit(X_train_tensor, y_train_tensor)
        model = model

    torch_save_model(
        {
            "model_state_dict": model.module_.state_dict(),
            "vocab": vocab,
            "model_class": lambda: SimpleNet()
        },
        "./TrainedModels/pytorchClassifier.pth"
    )
    print("PyTorch SimpleNet (GloVe) Finished")
    return model, vocab


def pytorchDeepClassifier(X_text, y_encoded, do_grid_search=True):
    print("\nPyTorch Deep MLP (GloVe) Started")

    le = LabelEncoder()
    y = le.fit_transform(y_encoded).astype(np.int64)

    X_padded, vocab, embedding_matrix = build_glove_embeddings(X_text)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_padded, y, test_size=0.2, stratify=y, random_state=1103)

    num_classes = len(np.unique(y_train))
    embed_dim = embedding_matrix.shape[1]

    class DeepMLPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
            self.net = nn.Sequential(
                nn.Linear(embed_dim*2, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.6),
                nn.Linear(256, num_classes)
            )

        def forward(self, X):
            emb = self.embedding(X)
            mean_emb = emb.mean(dim=1)
            max_emb = emb.max(dim=1).values
            features = torch.cat([mean_emb, max_emb], dim=1)  # double embedding info
            return self.net(features)
        
    grad_clip = GradientNormClipping(1.0)
    lr_scheduler = LRScheduler(
    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
    monitor='valid_loss',      # what to monitor
    patience=2,
    factor=0.5,
    )
    early_stop = EarlyStopping(patience=3, monitor='valid_loss', lower_is_better=True, load_best=True)
    
    model = NeuralNetClassifier(
        module=DeepMLPNet,
        max_epochs=20,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        criterion=nn.CrossEntropyLoss,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=3,
        train_split=lambda ds, y: (
            torch.utils.data.TensorDataset(X_train, torch.tensor(y_train)),
            torch.utils.data.TensorDataset(X_valid, torch.tensor(y_valid)),
        ),
        callbacks=[early_stop, grad_clip, lr_scheduler],
    )

    X_train_tensor = X_train
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    if do_grid_search:
        param_grid = {
            "lr": [0.0005, 0.0003, 0.0001],
            "batch_size": [256, 512, 1024]
        }
        gs = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=1, verbose=3)
        gs.fit(X_train_tensor, y_train_tensor)
        model = gs.best_estimator_
        print("Best parameters:", gs.best_params_)
    else:
        model.fit(X_train_tensor, y_train_tensor)
        model = model

    torch_save_model(
        {
            "model_state_dict": model.module_.state_dict(),
            "vocab": vocab,
            "model_class": lambda: DeepMLPNet()
        }, "./TrainedModels/pytorchDeepClassifier.pth")
    print("PyTorch Deep MLP (GloVe) Finished")
    return model, vocab


def pytorchLSTMClassifier(X_text, y_encoded, do_grid_search=True):
    print("\nPyTorch LSTM Classifier (GloVe + GPU) Started")
    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_encoded).astype(np.int64)

    # Use shared embedding builder
    X_padded, vocab, embedding_matrix = build_glove_embeddings(X_text)

    # Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_padded, y, test_size=0.2, stratify=y, random_state=1103
    )

    num_classes = len(np.unique(y_train))
    embed_dim = embedding_matrix.shape[1]
    hidden_dim = 128

    # Define LSTM model using embeddings
    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, X):
            emb = self.embedding(X)
            lstm_out, (h_n, _) = self.lstm(emb)
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # concat final states (bidirectional)
            out = self.dropout(out)
            return self.fc(out)

    # Early stopping
    early_stop = EarlyStopping(patience=3, monitor='valid_loss', lower_is_better=True, load_best=True)

    # Wrap with Skorch
    model = NeuralNetClassifier(
        module=LSTMNet,
        max_epochs=20,
        optimizer=torch.optim.AdamW,
        criterion=nn.CrossEntropyLoss,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=3,
        train_split=lambda ds, y: (
            TensorDataset(X_train, torch.tensor(y_train)),
            TensorDataset(X_valid, torch.tensor(y_valid))
        ),
        callbacks=[early_stop]
    )

    # Optional grid search
    if do_grid_search:
        param_grid = {
            "lr": [0.001, 0.0005, 0.0001],
            "batch_size": [256, 512, 1024]
        }
        gs = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=1, refit=True, verbose=3)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print("Best parameters LSTM (GloVe):", gs.best_params_)
    else:
        model.fit(X_train, y_train)

    torch_save_model(
        {
            "model_state_dict": model.module_.state_dict(),
            "vocab": vocab,
            "model_class": lambda: LSTMNet()
        }, "./TrainedModels/pytorchLSTMClassifier.pth")
    print("PyTorch LSTM Classifier (GloVe) Finished")

    return model, vocab


def pytorchCNNClassifier(X_text, y_encoded, do_grid_search=True):
    print("\nPyTorch CNN (GloVe) Started")

    le = LabelEncoder()
    y = le.fit_transform(y_encoded).astype(np.int64)
    
    X_padded, vocab, embedding_matrix = build_glove_embeddings(X_text)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_padded, y, test_size=0.2, stratify=y, random_state=1103)

    num_classes = len(np.unique(y_train))
    embed_dim = embedding_matrix.shape[1]

    class CNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
            self.conv = nn.Sequential(
                nn.Conv1d(embed_dim, 64, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(64, 32, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Linear(32, num_classes)

        def forward(self, X):
            emb = self.embedding(X)        # (batch, seq_len, embed_dim)
            emb = emb.transpose(1, 2)      # (batch, embed_dim, seq_len)
            feat = self.conv(emb).squeeze(-1)
            return self.fc(feat)

    early_stop = EarlyStopping(patience=3, monitor='valid_loss', lower_is_better=True, load_best=True)
    model = NeuralNetClassifier(
        module=CNNNet,
        max_epochs=20,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-4,
        criterion=nn.CrossEntropyLoss,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1,
        train_split=lambda ds, y: (
            torch.utils.data.TensorDataset(X_train, torch.tensor(y_train)),
            torch.utils.data.TensorDataset(X_valid, torch.tensor(y_valid))
        ),
        callbacks=[early_stop]
    )

    if do_grid_search:
        gs = GridSearchCV(model, {
            "lr": [0.001, 0.0005, 0.0001], 
            "batch_size": [256, 512, 1024]},
            scoring='accuracy', 
            n_jobs=1, 
            verbose=3)
        
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print("Best parameters CNN:", gs.best_params_)
    else:
        model.fit(X_train, y_train)

    torch_save_model(
        {
            "model_state_dict": model.module_.state_dict(),
            "vocab": vocab,
            "model_class": lambda: CNNNet()
        }, "./TrainedModels/pytorchCNNClassifier.pth")
    print("PyTorch CNN (GloVe) Finished")
    print(pd.DataFrame(gs.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
])
    return model, vocab


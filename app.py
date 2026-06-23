import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib
import io
import re

app = Flask(__name__)
CORS(app)

print("App started")

df_global = None
pipeline_global = None
df_processed_global = None
best_model_global = None
X_train_global = None
X_test_global = None
y_train_global = None
y_test_global = None

X_train_raw_global = None
X_test_raw_global  = None
y_train_raw_global = None
preprocessor_global     = None  
imbalance_apply_global  = False  

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# Upload Dataset
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload():
    global df_global
    file = request.files["file"]
    if file.filename.endswith(".csv"):
        df_global = pd.read_csv(file)
    else:
        df_global = pd.read_excel(file)
    preview_df = df_global.head(15).copy().replace({np.nan: None})
    columns = list(df_global.columns)
    categorical_cols = list(df_global.select_dtypes(include=["object"]).columns)
    return jsonify({
        "columns": columns,
        "rows": preview_df.values.tolist(),
        "categorical_cols": categorical_cols
    })

# ---------------------------
# Preprocessing
# ---------------------------
@app.route("/preprocess", methods=["POST"])
def preprocess():
    global df_global, pipeline_global, df_processed_global, X_train_global, X_test_global, y_train_global, y_test_global
    global X_train_raw_global, X_test_raw_global, y_train_raw_global, preprocessor_global, imbalance_apply_global
    if df_global is None:
        return jsonify({"error": "No dataset uploaded"}), 400
    
    data = request.get_json()
    task = data.get("task")
    target_col = data.get("target_column")
    ordinal_cols = data.get("ordinal_cols", [])
    
    df = df_global.copy()
    
    # ===============================
    # 1. REMOVE HIGH CARDINALITY COLUMNS
    # ===============================
    high_cardinality_cols = [
        col for col in df.columns
        if df[col].nunique() / len(df) > 0.5 and col != target_col
        and not pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]
    df = df.drop(columns=high_cardinality_cols, errors="ignore")
    
    print("High cardinality cols found:", high_cardinality_cols)
    print("Dtypes at this point:")
    print(df.dtypes)
    
    # ===============================
    # 2. REMOVE USELESS COLUMNS
    # ===============================
    id_cols = [
        col for col in df.columns
        if re.fullmatch(r'id', col.strip().lower()) or
        re.match(r'^id[_\s]', col.lower()) or
        re.search(r'[_\s]id$', col.lower()) or
        re.search(r'[_\s]id[_\s]', col.lower()) or
        re.search(r'(?<=[a-z])Id$', col)
    ]
    df = df.drop(columns=id_cols, errors="ignore")
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols, errors="ignore")
    
    # ===============================
    # 3. DROP HIGH MISSING
    # ===============================
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio >= 0.5].index.tolist()
    df = df.drop(columns=high_missing_cols, errors="ignore")
    
    # ===============================
    # 4. SPLIT FEATURES AND TARGET
    # ===============================
    imbalance_status = "Not Applied"
    if (task == "classification" or task == "regression") and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train = df.copy()
        X_test = pd.DataFrame(columns=df.columns)
        y_train = None
        y_test = None

    # Store cleaned data 
    X_train_raw_global = X_train.copy()
    X_test_raw_global  = X_test.copy()
    y_train_raw_global = y_train.copy() if y_train is not None else None
    y_test_global      = y_test.copy()  if y_test  is not None else None

    # ===============================
    # 5. DEFINE PIPELINE  
    # ===============================
    numeric_features    = X_train.select_dtypes(include=["number"]).columns.tolist()
    all_categorical     = X_train.select_dtypes(include=["object"]).columns.tolist()
    ordinal_cols_valid  = [c for c in ordinal_cols if c in all_categorical]
    nominal_cols        = [c for c in all_categorical if c not in ordinal_cols_valid]

    # Numeric sub-pipeline: impute with median → standardise
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    # Categorical sub-pipeline: impute with mode → one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers_list = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, nominal_cols),
    ]

    # Optional ordinal sub-pipeline when user selects ordinal columns
    if ordinal_cols_valid:
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())
        ])
        transformers_list.append(('ord', ordinal_transformer, ordinal_cols_valid))

    preprocessor = ColumnTransformer(transformers=transformers_list)
    preprocessor_global = preprocessor   

    # ===============================
    # 6. FIT PREPROCESSOR ON TRAIN 
    # ===============================
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr  = (preprocessor.transform(X_test)
                   if not X_test.empty
                   else np.zeros((0, X_train_arr.shape[1])))

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_arr.shape[1])]

    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names,
                            index=(X_test.index if not X_test.empty else range(0)))

    # ===============================
    # 7. SMOTE (ONLY IF CLASSIFICATION)
    # ===============================
    imbalance_apply_global = False
    if task == "classification" and y_train is not None:
        class_ratio = y_train.value_counts(normalize=True)
        min_class_count = y_train.value_counts().min()
        if class_ratio.min() < 0.3 and len(X_train) < 100000 and min_class_count >= 6:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            imbalance_status = "SMOTE Applied"
            imbalance_apply_global = True
        else:
            imbalance_status = "Already Balanced"
    else:
        imbalance_status = "Not Applied"

    # ===============================
    # 8. SAVE PIPELINE
    # ===============================
    pipeline_global = {
        "preprocessor":          preprocessor,        
        "high_cardinality_cols": high_cardinality_cols,
        "id_cols":               id_cols,
        "constant_cols":         constant_cols,
        "high_missing_cols":     high_missing_cols,
        "task":                  task,
        "target_col":            target_col
    }
    
    # ===============================
    # FINAL
    # ===============================
    # Reassemble full processed dataframes for preview and global storage
    df_train = X_train.copy()
    if y_train is not None:
        df_train[target_col] = y_train.values if hasattr(y_train, 'values') else y_train
    
    df_test = X_test.copy()
    if y_test is not None:
        df_test[target_col] = y_test.values
    
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    preview_df = df.head(15).replace({np.nan: None})
    
    df_global = df.copy()
    df_processed_global = df.copy()
    X_train_global = X_train.copy()
    X_test_global = X_test.copy()
    y_train_global = y_train.copy() if y_train is not None else None
    
    return jsonify({
        "columns": list(df.columns),
        "rows": preview_df.values.tolist(),
        "steps": {
            "removed_high_cardinality_cols": high_cardinality_cols,
            "removed_id_columns": id_cols,
            "removed_constant_columns": constant_cols,
            "removed_high_missing_columns": high_missing_cols,
            "numeric_imputation": "median (SimpleImputer)",
            "categorical_imputation": "most_frequent (SimpleImputer)",
            "ordinal_encoding": ordinal_cols_valid,
            "frequency_encoding": nominal_cols,
            "scaling": "StandardScaler (Pipeline)",
            "imbalance_handling": imbalance_status
        }
    })

# ---------------------------
# Train
# ---------------------------
@app.route("/train", methods=["POST"])
def train():
    global df_global, best_model_global, df_processed_global, X_train_global, X_test_global, y_train_global, y_test_global
    global X_train_raw_global, X_test_raw_global, y_train_raw_global, preprocessor_global, imbalance_apply_global
    
    if df_processed_global is None:
        return jsonify({"error": "Please run preprocessing first"}), 400
    if df_global is None:
        return jsonify({"error": "No dataset uploaded"}), 400
    
    data = request.get_json()
    task = data.get("task")
    target_col = data.get("target_column")
    
    results = {}
    trained_models = {}
    
    # ===============================
    # CLASSIFICATION
    # ===============================
    if task == "classification":
        X_train = X_train_raw_global.copy()
        X_test  = X_test_raw_global.copy()
        y_train = y_train_raw_global.copy()
        y_test  = y_test_global.copy()
        
        if y_train.nunique() < 2:
            return jsonify({"error": "Target must have at least 2 classes"}), 400
        if y_train.nunique() > 20:
            return jsonify({"error": "Too many unique values for classification"}), 400
        
        classifiers = {
            "Logistic Regression": (
                LogisticRegression(random_state=42, max_iter=1000),
                {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"]
                }
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                }
            )
        }
        
        # Train + Validate (preprocessor + classifier)
        for name, (clf, param_grid) in classifiers.items():
            # Build full pipeline
            if imbalance_apply_global:
                model_pipeline = ImbPipeline(steps=[
                    ('preprocessor', clone(preprocessor_global)),
                    ('smote',        SMOTE(random_state=42)),
                    ('classifier',   clf)
                ])
            else:
                model_pipeline = Pipeline(steps=[
                    ('preprocessor', clone(preprocessor_global)),
                    ('classifier',   clf)
                ])

            prefixed_params = {f'classifier__{k}': v for k, v in param_grid.items()}
            search = RandomizedSearchCV(
                model_pipeline, prefixed_params,
                n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_   # full pipeline
            y_pred = best_estimator.predict(X_test)
            results[name] = accuracy_score(y_test, y_pred)
            trained_models[name] = best_estimator
        
        # Pick Best Model
        best_name = max(results, key=results.get)
        best_model_global = trained_models[best_name]
        
        # Final Test Evaluation
        y_pred_best = best_model_global.predict(X_test)
        metrics = {
            "best_model": best_name,
            "all_scores": {k: round(v, 4) for k, v in results.items()},
            "accuracy": round(accuracy_score(y_test, y_pred_best), 4),
            "precision": round(precision_score(y_test, y_pred_best, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_best, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred_best, average="weighted", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_best).tolist()
        }
    
    # ===============================
    # REGRESSION
    # ===============================
    elif task == "regression":
        X_train = X_train_raw_global.copy()
        X_test  = X_test_raw_global.copy()
        y_train = y_train_raw_global.copy()
        y_test  = y_test_global.copy()
        
        models = {
            "Linear Regression": (
                LinearRegression(),
                {
                    "fit_intercept": [True, False]
                }
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                }
            )
        }
        
        # Train + Validate (preprocessor + regressor)
        for name, (reg, param_grid) in models.items():
            model_pipeline = Pipeline(steps=[
                ('preprocessor', clone(preprocessor_global)),
                ('regressor',    reg)
            ])
            prefixed_params = {f'regressor__{k}': v for k, v in param_grid.items()}
            search = RandomizedSearchCV(
                model_pipeline, prefixed_params,
                n_iter=10, cv=3, scoring="r2", random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_  
            results[name] = search.best_score_
            trained_models[name] = best_estimator
        
        # Pick Best Model
        best_name = max(results, key=results.get)
        best_model_global = trained_models[best_name]
        
        # Final Test Evaluation
        y_pred_test = best_model_global.predict(X_test)
        metrics = {
            "best_model": best_name,
            "all_scores": {k: round(v, 4) for k, v in results.items()},
            "MAE": round(mean_absolute_error(y_test, y_pred_test), 4),
            "MSE": round(mean_squared_error(y_test, y_pred_test), 4),
            "R2": round(r2_score(y_test, y_pred_test), 4)
        }
    
    # ===============================
    # CLUSTERING
    # ===============================
    elif task == "clustering":
        X = df_processed_global.copy()
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category").cat.codes
        
        # Find best k using silhouette score
        best_k = 2
        best_sil = -1
        for k in range(2, 11):
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_sil:
                best_sil = score
                best_k = k
        
        models = {
            "KMeans": KMeans(n_clusters=best_k, random_state=42),
            "Hierarchical Clustering": AgglomerativeClustering(n_clusters=best_k)
        }
        
        for name, model in models.items():
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            results[name] = score
            trained_models[name] = model
        
        best_name = max(results, key=results.get)
        best_model_global = trained_models[best_name]
        
        metrics = {
            "best_model": best_name,
            "all_scores": results,
            "silhouette_score": results[best_name],
            "best_k": best_k
        }
    
    return jsonify({"task": task, "metrics": metrics})

# ---------------------------
# Download Model + Pipeline
# ---------------------------
@app.route("/download-model", methods=["GET"])
def download_model():
    global best_model_global
    if best_model_global is None:
        return jsonify({"error": "No model trained yet"}), 400
    
    #full Pipeline (preprocessor + model),
    buffer = io.BytesIO()
    joblib.dump(best_model_global, buffer)
    buffer.seek(0)
    
    return send_file(
        buffer,
        download_name="model_pipeline.pkl",
        as_attachment=True,
        mimetype="application/octet-stream"
    )

if __name__ == "__main__":
    app.run(debug=True)

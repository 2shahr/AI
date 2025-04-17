import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('kdd_train.csv')
test_df = pd.read_csv('kdd_test.csv')

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

train_df['labels'] = train_df['labels'].apply(lambda x: 1 if x.strip().lower() == 'normal' else 0)
test_df['labels'] = test_df['labels'].apply(lambda x: 1 if x.strip().lower() == 'normal' else 0)

X_train = train_df.drop('labels', axis=1)
y_train = train_df['labels']
X_test = test_df.drop('labels', axis=1)
y_test = test_df['labels']

categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', MinMaxScaler(), numerical_cols)
])

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42)
}

trained_models = {}
transformed_data = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('pre', preprocessor), ('clf', model)])
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
    acc = pipeline.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, pipeline.predict(X_test)))

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

for name in models:
    print(f"\nSHAP Summary for {name}")
    model = trained_models[name].named_steps['clf']
    explainer = shap.Explainer(model, X_train_transformed)
    shap_values = explainer(X_test_transformed[:100])
    shap.summary_plot(shap_values, X_test_transformed[:100], feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary - {name}")
    plt.show()

for name in models:
    print(f"\nPermutation Importance for {name}")
    model = trained_models[name].named_steps['clf']
    
    result = permutation_importance(
        model,
        X_test_transformed,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring='accuracy'
    )

    sorted_idx = result.importances_mean.argsort()[-15:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], xerr=result.importances_std[sorted_idx])
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel("Mean Importance Decrease")
    plt.title(f"Top 15 Permutation Importances - {name}")
    plt.tight_layout()
    plt.show()

import shap
sample_idx = 10

for name in models:
    print(f"\nSHAP Waterfall for {name} (sample {sample_idx})")
    
    model = trained_models[name].named_steps['clf']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_val = expected_value[1]
        shap_val = shap_values[1][sample_idx]
    else:
        expected_val = expected_value
        shap_val = shap_values[sample_idx]

    shap_exp = shap.Explanation(
        values=shap_val,
        base_values=expected_val,
        data=X_test_transformed[sample_idx],
        feature_names=feature_names
    )

    shap.plots.waterfall(shap_exp, max_display=10)

from IPython.display import display

print("\nLIME Local Explanation (Sample index =", sample_idx, ")")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed,
    feature_names=feature_names,
    class_names=["Attack", "Normal"],
    discretize_continuous=True,
    verbose=True,
    random_state=42
)

for name in models:
    print(f"\nLIME explanation for {name}:")
    model = trained_models[name].named_steps['clf']
    
    def predict_fn(x):
        return model.predict_proba(x)
    instance = X_test_transformed[sample_idx].toarray()[0] if hasattr(X_test_transformed, "toarray") else X_test_transformed[sample_idx]
    lime_exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn
    )
    
    display(lime_exp.show_in_notebook(show_table=True, show_all=False))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/datasets/openml-datasets/main/data/credit-g/credit-g.csv"

st.set_page_config(
    page_title="Credit Scoring Trustworthy AI",
    page_icon="🤖",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load German Credit dataset from public repository."""
    return pd.read_csv(DATA_URL)


def get_feature_names_from_column_transformer(column_transformer: ColumnTransformer) -> np.ndarray:
    feature_names = []

    for name, transformer, columns in column_transformer.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
        else:
            names = columns

        feature_names.extend(names)

    return np.array(feature_names)


@st.cache_resource(show_spinner=True)
def train_models():
    df = load_data().copy()
    target_col = "class"

    df["target"] = df[target_col].map({"good": 0, "bad": 1})

    if "age" in df.columns:
        df["age_group"] = np.where(df["age"] < df["age"].median(), "younger", "older")
    else:
        df["age_group"] = "unknown"

    sensitive_col = "personal_status" if "personal_status" in df.columns else "age_group"

    X = df.drop(columns=[target_col, "target"])
    y = df["target"]
    sensitive_features = df[sensitive_col]

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive_features,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=7,
        ),
    }

    trained_models = {}
    results = []

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        trained_models[model_name] = pipeline
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": f1_score,
            "selection_rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred_best,
        sensitive_features=sensitive_test,
    )

    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(
        y_test,
        y_pred_best,
        target_names=["Good Risk", "Bad Risk"],
        zero_division=0,
        output_dict=True,
    )

    feature_importance_df = pd.DataFrame()
    classifier = best_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        feature_names = get_feature_names_from_column_transformer(best_model.named_steps["preprocess"])
        feature_importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": classifier.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(15)
        )

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_test": sensitive_test,
        "sensitive_col": sensitive_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "results_df": results_df,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "y_pred_best": y_pred_best,
        "metric_frame": metric_frame,
        "fairness_df": metric_frame.by_group,
        "cm": cm,
        "report": report,
        "feature_importance_df": feature_importance_df,
    }


def metric_card(label: str, value: str):
    st.metric(label=label, value=value)


st.title("Implementasi Credit Scoring Berbasis Trustworthy AI")
st.caption("Versi web app interaktif untuk mendampingi notebook presentasi-friendly.")

with st.spinner("Menyiapkan dataset dan melatih model..."):
    artifacts = train_models()


df = artifacts["df"]
X = artifacts["X"]
results_df = artifacts["results_df"]
best_model_name = artifacts["best_model_name"]
best_model = artifacts["best_model"]
fairness_df = artifacts["fairness_df"]
metric_frame = artifacts["metric_frame"]

st.sidebar.header("Navigasi")
page = st.sidebar.radio(
    "Pilih bagian:",
    [
        "Ringkasan Proyek",
        "Dataset & Pipeline",
        "Performa Model",
        "Fairness",
        "Simulasi Prediksi",
        "Trustworthy AI & SDGs",
    ],
)

st.sidebar.info(
    "Catatan: Aplikasi ini adalah demo edukatif untuk presentasi tugas AI, bukan sistem keputusan kredit nyata."
)

if page == "Ringkasan Proyek":
    st.header("Ringkasan Proyek")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Jumlah data", f"{df.shape[0]:,}")
    with col2:
        metric_card("Jumlah fitur", f"{X.shape[1]:,}")
    with col3:
        metric_card("Model terbaik", best_model_name)
    with col4:
        best_f1 = results_df.loc[results_df["Model"] == best_model_name, "F1-score"].iloc[0]
        metric_card("F1-score terbaik", f"{best_f1:.3f}")

    st.subheader("Tujuan")
    st.write(
        "Aplikasi ini menunjukkan alur implementasi credit scoring berbasis machine learning, "
        "mulai dari data understanding, preprocessing, training, evaluasi performa, evaluasi fairness, "
        "hingga interpretasi dalam konteks Trustworthy AI."
    )

    st.subheader("Alur Pipeline")
    pipeline_df = pd.DataFrame(
        [
            ["1", "Load Dataset", "Mengambil German Credit Data sebagai data mentah."],
            ["2", "Data Understanding", "Mengecek struktur data, missing value, dan distribusi target."],
            ["3", "Preprocessing", "StandardScaler untuk numerik dan OneHotEncoder untuk kategorikal."],
            ["4", "Training", "Membandingkan Logistic Regression, Decision Tree, dan Random Forest."],
            ["5", "Evaluasi", "Menggunakan accuracy, precision, recall, dan F1-score."],
            ["6", "Fairness", "Membandingkan performa berdasarkan atribut sensitif."],
            ["7", "Interpretasi", "Melihat feature importance untuk transparansi model."],
        ],
        columns=["No", "Pipeline", "Penjelasan"],
    )
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

elif page == "Dataset & Pipeline":
    st.header("Dataset & Pipeline")
    st.subheader("Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Target")
        target_counts = df["target"].value_counts().sort_index()
        target_view = pd.DataFrame(
            {
                "Kelas": ["Good credit risk (0)", "Bad credit risk (1)"],
                "Jumlah": target_counts.values,
                "Persentase": (target_counts / target_counts.sum() * 100).round(2).values,
            }
        )
        st.dataframe(target_view, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Jenis Fitur")
        st.write("**Fitur numerik:**")
        st.write(", ".join(artifacts["numeric_features"]))
        st.write("**Fitur kategorikal:**")
        st.write(", ".join(artifacts["categorical_features"]))

    st.subheader("Penjelasan Pipeline Preprocessing")
    st.write(
        "Pipeline preprocessing memisahkan fitur numerik dan kategorikal. Fitur numerik dinormalisasi "
        "dengan StandardScaler, sedangkan fitur kategorikal diubah menjadi representasi numerik menggunakan "
        "OneHotEncoder. Semua proses digabungkan dalam ColumnTransformer agar rapi dan konsisten."
    )

elif page == "Performa Model":
    st.header("Performa Model")
    st.subheader("Perbandingan Metrik")
    st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)

    st.subheader("Visualisasi F1-score")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results_df["Model"], results_df["F1-score"])
    ax.set_title("Perbandingan F1-score Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1-score")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    st.subheader("Confusion Matrix Model Terbaik")
    cm = artifacts["cm"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Good", "Actual Bad"],
        columns=["Predicted Good", "Predicted Bad"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(artifacts["report"]).T
    st.dataframe(report_df.round(4), use_container_width=True)

elif page == "Fairness":
    st.header("Evaluasi Fairness")
    st.write(f"Atribut sensitif yang digunakan: **{artifacts['sensitive_col']}**")

    st.subheader("Metrik per Kelompok Sensitif")
    st.dataframe(fairness_df.round(4), use_container_width=True)

    st.subheader("Visualisasi Metrik Fairness")
    fig, ax = plt.subplots(figsize=(10, 5))
    fairness_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Evaluasi Fairness berdasarkan {artifacts['sensitive_col']}")
    ax.set_xlabel("Kelompok Sensitif")
    ax.set_ylabel("Nilai Metrik")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Difference")
        st.dataframe(metric_frame.difference(method="between_groups").to_frame("difference").round(4))
    with col2:
        st.subheader("Ratio")
        st.dataframe(metric_frame.ratio(method="between_groups").to_frame("ratio").round(4))

    st.info(
        "Interpretasi: semakin besar perbedaan antar kelompok, semakin perlu dilakukan investigasi bias. "
        "Evaluasi ini masih sederhana dan ditujukan untuk pembelajaran Trustworthy AI."
    )

elif page == "Simulasi Prediksi":
    st.header("Simulasi Prediksi Risiko Kredit")
    st.write(
        "Gunakan form berikut untuk mencoba prediksi satu calon peminjam. Nilai awal diambil dari salah satu baris dataset, "
        "lalu bisa diedit sesuai kebutuhan presentasi."
    )

    selected_idx = st.slider("Pilih contoh baris dataset", 0, len(X) - 1, 0)
    sample = X.iloc[selected_idx].copy()

    with st.form("prediction_form"):
        input_values = {}
        numeric_features = artifacts["numeric_features"]
        categorical_features = artifacts["categorical_features"]

        st.subheader("Fitur Numerik")
        num_cols = st.columns(3)
        for i, col in enumerate(numeric_features):
            value = float(sample[col])
            min_value = float(X[col].min())
            max_value = float(X[col].max())
            step = 1.0 if np.issubdtype(X[col].dtype, np.integer) else 0.1
            input_values[col] = num_cols[i % 3].number_input(
                col,
                min_value=min_value,
                max_value=max_value,
                value=value,
                step=step,
            )

        st.subheader("Fitur Kategorikal")
        cat_cols = st.columns(2)
        for i, col in enumerate(categorical_features):
            options = sorted(X[col].dropna().astype(str).unique().tolist())
            current = str(sample[col])
            index = options.index(current) if current in options else 0
            input_values[col] = cat_cols[i % 2].selectbox(col, options=options, index=index)

        submitted = st.form_submit_button("Prediksi Risiko Kredit")

    if submitted:
        input_df = pd.DataFrame([input_values])
        pred = best_model.predict(input_df)[0]
        label = "Bad Risk" if pred == 1 else "Good Risk"
        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi model: **{label}**")

        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(input_df)[0]
            st.write(f"Probabilitas Good Risk: **{proba[0]:.2%}**")
            st.write(f"Probabilitas Bad Risk: **{proba[1]:.2%}**")

        st.caption(
            "Catatan: hasil ini hanya simulasi model edukatif. Dalam implementasi nyata, keputusan kredit perlu audit, validasi, "
            "penjelasan keputusan, dan kepatuhan regulasi."
        )

elif page == "Trustworthy AI & SDGs":
    st.header("Trustworthy AI, Regulasi, dan SDGs")

    st.subheader("Fairness")
    st.write(
        "Model credit scoring harus diperiksa apakah performanya berbeda secara signifikan antar kelompok sensitif. "
        "Perbedaan besar dapat menjadi indikasi potensi bias."
    )

    st.subheader("Transparency")
    st.write(
        "Pipeline, metrik, dan interpretasi model perlu dijelaskan agar keputusan model tidak dianggap sebagai black box."
    )

    st.subheader("Accountability")
    st.write(
        "Setiap tahap, mulai dari sumber data, preprocessing, pemilihan model, evaluasi, hingga interpretasi perlu terdokumentasi."
    )

    st.subheader("Ethics")
    st.write(
        "Credit scoring berdampak pada akses keuangan seseorang. Karena itu, sistem AI harus menghindari diskriminasi dan tetap manusiawi."
    )

    st.subheader("Kaitan dengan SDGs")
    sdg_df = pd.DataFrame(
        [
            ["SDG 8", "Decent Work and Economic Growth", "Sistem kredit yang adil dapat mendukung akses pembiayaan yang lebih inklusif."],
            ["SDG 10", "Reduced Inequalities", "Evaluasi fairness membantu mengurangi potensi diskriminasi antar kelompok."],
        ],
        columns=["SDG", "Tema", "Kaitan"],
    )
    st.dataframe(sdg_df, use_container_width=True, hide_index=True)

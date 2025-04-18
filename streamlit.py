import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(layout="wide")
st.title("📊 NHANES Insurance Prediction and Health Data Exploration")

@st.cache_data
def load_dataset(file):
    return pd.read_sas(file, format="xport")

# Sidebar file uploads
st.sidebar.header("Upload NHANES .xpt Files")
files = {}
for label in ["BPXO_L", "DEMO_L", "FERTIN_L", "HIQ_L", "HSCRP_L", "KIQ_U_L", "BMX_L", "VID_L", "UCPREG_L"]:
    uploaded_file = st.sidebar.file_uploader(f"Upload {label}.xpt", type="xpt")
    if uploaded_file:
        files[label] = load_dataset(uploaded_file)

# Data processing and merging
if len(files) >= 3:
    st.subheader("✅ Loaded Datasets")
    for name, df in files.items():
        st.write(f"**{name}**")
        st.dataframe(df.head())

    # Merge datasets on SEQN
    data = files["DEMO_L"]
    for name, df in files.items():
        if name != "DEMO_L":
            data = pd.merge(data, df, on="SEQN", how="inner")

    data = data.drop_duplicates(subset="SEQN")
    data.columns = data.columns.str.strip().str.lower()

    # Drop columns with >50% missing
    missing_percent = data.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > 0.5].index.tolist()
    data = data.drop(columns=cols_to_drop)

    # Impute numeric columns
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    num_imputer = SimpleImputer(strategy="median")
    data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

    # Filter females only (if gender present)
    if "riagendr" in data.columns:
        data = data[data["riagendr"] != 1.0]

    # Target: Insurance-related columns
    target_cols = [col for col in data.columns if col.startswith("hiq")]

    st.markdown("### 🎯 Insurance-related Target Columns")
    st.write(target_cols)

    # Define features and labels
    x = data.drop(columns=target_cols, errors="ignore")
    y = data[target_cols]

    if not y.empty:
        # Preprocessing
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x.select_dtypes(include=["int64", "float64"]))

        poly = PolynomialFeatures(degree=2, include_bias=False)
        x_poly = poly.fit_transform(x_scaled)

        x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.3, random_state=0)

        # Train logistic regression
        base_model = LogisticRegression(max_iter=1000)
        model = MultiOutputClassifier(base_model)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        st.markdown("### 📈 Accuracy per Target")
        for i, col in enumerate(y.columns):
            acc = accuracy_score(y_test[col], y_pred[:, i])
            st.write(f"**{col}: {acc:.4f}**")

        st.markdown("### 🧾 Classification Reports")
        for i, col in enumerate(y.columns):
            report = classification_report(y_test[col], y_pred[:, i], output_dict=True, zero_division=1)
            st.write(f"**{col}**")
            st.json(report)

        
        # 🔥 Feature Importance via Random Forest
        st.markdown("### 🌲 Feature Importances (RandomForestClassifier)")

        numeric_input = x.select_dtypes(include=["int64", "float64"])
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(numeric_input, y[target_cols[0]])  # use first target for simplicity

        importances = rf.feature_importances_
        feature_names = numeric_input.columns

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)  

        # 📊 Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap of Numeric Features")
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        st.pyplot(fig)


        # 📦 Boxplots of numeric features
        st.markdown("### 📦 Boxplots of Numeric Features")

        # Select numeric features
        numeric_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Choose batch size
        batch_size = st.slider("Number of features per row:", min_value=3, max_value=10, value=5)

        # Calculate total batches
        num_batches = len(numeric_features) // batch_size + (len(numeric_features) % batch_size != 0)

        # Loop and show boxplots
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            features_batch = numeric_features[start:end]

            fig, axs = plt.subplots(1, len(features_batch), figsize=(5 * len(features_batch), 4))

            if len(features_batch) == 1:
                axs = [axs]

            for i, feature in enumerate(features_batch):
                sns.boxplot(y=x[feature], ax=axs[i])
                axs[i].set_title(feature.upper())
                axs[i].set_xlabel("")

            st.pyplot(fig)


    else:
        st.warning("No insurance-related columns found in merged data.")

else:
    st.info("Upload at least 3 .xpt NHANES files to begin.")

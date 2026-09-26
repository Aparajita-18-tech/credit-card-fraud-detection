import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning application using Logistic Regression.")

# ---------------------------------------------------------
# Load / upload dataset
# ---------------------------------------------------------
st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload creditcard.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded successfully!")
else:
    st.info("Please upload your creditcard.csv file from the sidebar.")
    st.stop()

# ---------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("First 5 Rows")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Dataset Information")
info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes],
    "Missing Values": df.isnull().sum().values
})
st.dataframe(info_df, use_container_width=True)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
if "Class" not in df.columns:
    st.error("The dataset must contain a 'Class' column.")
    st.stop()

if df.isnull().sum().sum() > 0:
    st.warning("The dataset contains missing values. Please clean the dataset before training.")

st.subheader("Duplicate Transactions")
st.write(f"Duplicate rows: **{df.duplicated().sum()}**")

# ---------------------------------------------------------
# Class distribution
# ---------------------------------------------------------
st.header("2. Fraud vs Legitimate Transactions")

class_counts = df["Class"].value_counts().sort_index()
class_percentage = df["Class"].value_counts(normalize=True).sort_index() * 100

class_table = pd.DataFrame({
    "Class": class_counts.index,
    "Number of Transactions": class_counts.values,
    "Percentage": class_percentage.round(4).values
})

st.dataframe(class_table, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Class", data=df, ax=ax)
ax.set_title("Legitimate vs Fraudulent Transactions")
ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
ax.set_ylabel("Number of Transactions")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------
st.header("3. Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Amount"], bins=50, ax=ax)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Transaction Amount by Class")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="Class", y="Amount", data=df, ax=ax)
    ax.set_title("Transaction Amount: Legitimate vs Fraud")
    ax.set_xlabel("Class")
    ax.set_ylabel("Amount")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Transaction Time Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["Time"], bins=50, ax=ax)
ax.set_title("Transaction Time Distribution")
ax.set_xlabel("Time")
ax.set_ylabel("Number of Transactions")
st.pyplot(fig)
plt.close(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", linewidths=0.1, ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------
st.header("4. Prepare Data")

X = df.drop("Class", axis=1)
y = df["Class"]

st.write(f"**X shape:** {X.shape}")
st.write(f"**y shape:** {y.shape}")

# ---------------------------------------------------------
# Train / test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

st.write(f"**Training data:** {X_train.shape}")
st.write(f"**Testing data:** {X_test.shape}")

# ---------------------------------------------------------
# Scaling
# ---------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scale_columns = [col for col in ["Time", "Amount"] if col in X.columns]

if scale_columns:
    X_train_scaled[scale_columns] = scaler.fit_transform(
        X_train[scale_columns]
    )
    X_test_scaled[scale_columns] = scaler.transform(
        X_test[scale_columns]
    )

# ---------------------------------------------------------
# Model training
# ---------------------------------------------------------
st.header("5. Logistic Regression Model")

if st.button("Train Logistic Regression Model", type="primary"):

    with st.spinner("Training model..."):
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = list(X.columns)

    # Save model and scaler
    joblib.dump(model, "fraud_detection_model.pkl")
    joblib.dump(scaler, "fraud_detection_scaler.pkl")

    st.success("Model training completed successfully!")
    st.info("Model and scaler have been saved as .pkl files.")

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    st.header("6. Model Evaluation")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{accuracy:.4f}")
    m2.metric("Precision", f"{precision:.4f}")
    m3.metric("Recall", f"{recall:.4f}")
    m4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# Load previously saved model
# ---------------------------------------------------------
st.header("7. Fraud Prediction")

if "model" not in st.session_state:
    try:
        st.session_state["model"] = joblib.load("fraud_detection_model.pkl")
        st.session_state["scaler"] = joblib.load("fraud_detection_scaler.pkl")

        st.session_state["feature_columns"] = list(X.columns)
        st.success("Previously saved model loaded successfully.")
    except FileNotFoundError:
        st.warning("Train the model first to use fraud prediction.")

if "model" in st.session_state:

    model = st.session_state["model"]
    saved_scaler = st.session_state["scaler"]
    feature_columns = st.session_state["feature_columns"]

    st.subheader("Select a Transaction")

    row_number = st.number_input(
        "Enter dataset row number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    if st.button("Predict Selected Transaction"):

        transaction = df.iloc[[int(row_number)]].drop("Class", axis=1)

        transaction_scaled = transaction.copy()

        scale_columns = [
            col for col in ["Time", "Amount"]
            if col in transaction_scaled.columns
        ]

        if scale_columns:
            transaction_scaled[scale_columns] = saved_scaler.transform(
                transaction[scale_columns]
            )

        prediction = model.predict(transaction_scaled)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(transaction_scaled)[0][1]
        else:
            probability = None

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        if probability is not None:
            st.write(
                f"Fraud probability: **{probability * 100:.2f}%**"
            )

        st.dataframe(transaction, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Credit Card Fraud Detection using Machine Learning")

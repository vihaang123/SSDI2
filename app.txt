import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
 
# Classification models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
 
# Regression
from sklearn.linear_model import LinearRegression
 
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
 
st.set_page_config(page_title="ML Studio", layout="wide")
 
st.title("📊 Data Analysis & Machine Learning Studio")
 
# Upload dataset
file = st.file_uploader("Upload CSV file", type=["csv"])
 
if file is not None:
    df = pd.read_csv(file)
 
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
    # =================================================
    # 🔥 MULTI GRAPH MACRO (NO BAR PLOTS)
    # =================================================
    def show_multi_graphs(df, numeric_cols):
        st.subheader("📊 Multi Graph Visualization")
 
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available.")
            return
 
        cols = numeric_cols[:4]   # limit for performance
 
        # ---------------- HISTOGRAMS ----------------
        st.write("### Distribution Graphs")
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.flatten()
 
        for i, col in enumerate(cols):
            axes[i].hist(df[col], bins=20)
            axes[i].set_title(col)
 
        plt.tight_layout()
        st.pyplot(fig)
 
        # ---------------- LINE GRAPHS ----------------
        st.write("### Line Graphs")
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6))
        axes2 = axes2.flatten()
 
        for i, col in enumerate(cols):
            axes2[i].plot(df[col])
            axes2[i].set_title(col)
 
        plt.tight_layout()
        st.pyplot(fig2)
 
        # ---------------- SCATTER ----------------
        if len(cols) >= 2:
            st.write("### Scatter Relationship")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.scatter(df[cols[0]], df[cols[1]])
            ax3.set_xlabel(cols[0])
            ax3.set_ylabel(cols[1])
            ax3.set_title(f"{cols[0]} vs {cols[1]}")
            st.pyplot(fig3)
 
        # ---------------- BOXPLOTS ----------------
        st.write("### Boxplots (Outliers)")
        fig4, axes4 = plt.subplots(1, len(cols), figsize=(12, 4))
 
        if len(cols) == 1:
            axes4 = [axes4]
 
        for i, col in enumerate(cols):
            axes4[i].boxplot(df[col])
            axes4[i].set_title(col)
 
        plt.tight_layout()
        st.pyplot(fig4)
 
        # ---------------- HEATMAP ----------------
        if len(cols) > 1:
            st.write("### Correlation Heatmap")
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", ax=ax5)
            st.pyplot(fig5)
 
    # ==============================
    # CREATE TABS
    # ==============================
    tab1, tab2 = st.tabs(["📊 Data Analysis", "🤖 Machine Learning"])
 
    # =================================================
    # 📊 DATA ANALYSIS TAB
    # =================================================
    with tab1:
 
        st.header("Dataset Overview")
 
        st.subheader("Preview")
        st.dataframe(df.head())
 
        st.write(f"**Rows:** {df.shape[0]}   |   **Columns:** {df.shape[1]}")
 
        st.subheader("Data Types")
        st.write(df.dtypes)
 
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
 
        st.subheader("Statistical Summary")
        st.write(df.describe())
 
        st.markdown("---")
 
        if st.button("Generate Multi Graphs"):
            show_multi_graphs(df, numeric_cols)
 
    # =================================================
    # 🤖 MACHINE LEARNING TAB
    # =================================================
    with tab2:
 
        st.header("Model Builder")
 
        problem_type = st.radio(
            "Select Problem Type",
            ["Classification", "Regression"]
        )
 
        # select model first
        if problem_type == "Classification":
            model_choice = st.selectbox(
                "Select Classification Model",
                ["Gaussian NB", "Logistic Regression", "KNN", "SVM", "Decision Tree"]
            )
        else:
            model_choice = st.selectbox(
                "Select Regression Model",
                ["Linear Regression", "KNN Regressor", "SVR", "Decision Tree Regressor"]
            )
 
        # target column
        if problem_type == "Classification":
 
            possible_targets = categorical_cols.copy()
 
            for col in numeric_cols:
                if df[col].nunique() <= 10:
                    possible_targets.append(col)
 
            possible_targets = list(set(possible_targets))
 
            target_col = st.selectbox("Select Target Column", possible_targets)
 
        else:
            target_col = st.selectbox("Select Target Column", numeric_cols)
 
        # feature columns
        feature_cols = st.multiselect(
            "Select Feature Columns",
            [col for col in numeric_cols if col != target_col]
        )
 
        if len(feature_cols) == 0:
            st.warning("Select at least one feature.")
            st.stop()
 
        X = df[feature_cols]
        y = df[target_col]
 
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
 
        if st.button("Train Model"):
 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
 
            # ================= CLASSIFICATION =================
            if problem_type == "Classification":
 
                if model_choice == "Gaussian NB":
                    model = GaussianNB()
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_choice == "KNN":
                    model = KNeighborsClassifier()
                elif model_choice == "SVM":
                    model = SVC()
                else:
                    model = DecisionTreeClassifier()
 
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
 
                st.subheader("Accuracy")
                st.success(f"{accuracy_score(y_test, y_pred):.4f}")
 
                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred))
 
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
 
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
 
            # ================= REGRESSION =================
            else:
 
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "KNN Regressor":
                    model = KNeighborsRegressor()
                elif model_choice == "SVR":
                    model = SVR()
                else:
                    model = DecisionTreeRegressor()
 
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
 
                st.subheader("Regression Metrics")
                st.write("R² Score:", r2_score(y_test, y_pred))
                st.write("MAE:", mean_absolute_error(y_test, y_pred))
                st.write("MSE:", mean_squared_error(y_test, y_pred))
 
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
 
else:
    st.info("Upload a dataset to begin.")
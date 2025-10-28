"""
Using Telco Customer Churn data from the Kaggle.
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Resources:
https://www.kdnuggets.com/step-by-step-tutorial-to-building-your-first-machine-learning-model
https://www.kdnuggets.com/deploying-machine-learning-models-a-step-by-step-tutorial

original code and how to run: https://github.com/CornelliusYW/churn_prediction_machine_learning_development

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def statsInfo(df):
    print(df.head())

    # Get the basic information about the dataset
    print(df.info())

    # Get the numerical summary statistics of the dataset
    df.describe()

    # Get the categorical summary statistics of the dataset
    print(df.describe(exclude="number"))

    # Check for missing values
    print(df.isnull().sum())

    print(df["Churn"].value_counts())


def charts(df):
    df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype("str")

    df["ChurnTarget"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    num_features = df.select_dtypes("number").columns
    df[num_features].hist(bins=15, figsize=(15, 6), layout=(2, 5))

    # Plot distribution of categorical features
    cat_features = df.drop("customerID", axis=1).select_dtypes(include="object").columns

    plt.figure(figsize=(20, 20))
    for i, col in enumerate(cat_features, 1):
        plt.subplot(5, 4, i)
        df[col].value_counts().plot(kind="bar")
        plt.title(col)

    # Plot correlations between numerical features
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_features].corr())
    plt.title("Correlation Heatmap")
    plt.show()


from dython.nominal import associations


def correlationAnalysis(df):
    # Plot distribution of categorical features
    cat_features = df.drop("customerID", axis=1).select_dtypes(include="object").columns
    # Calculate the Cramer’s V and correlation matrix
    assoc = associations(df[cat_features], nominal_columns="all", plot=False)
    corr_matrix = assoc["corr"]

    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix)
    # plt.show()

    # Plot box plots to identify outliers
    plt.figure(figsize=(20, 15))
    num_features = df.select_dtypes("number").columns
    for i, col in enumerate(num_features, 1):
        plt.subplot(4, 4, i)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.show()


def featureselection(df):
    df["ChurnTarget"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    target = "ChurnTarget"
    num_features = df.select_dtypes(include=[np.number]).columns.drop(target)

    # Plot distribution of categorical features
    cat_features = df.drop("customerID", axis=1).select_dtypes(include="object").columns

    # Calculate correlations
    correlations = df[num_features].corrwith(df[target])

    # Set a threshold for feature selection
    threshold = 0.3
    selected_num_features = correlations[abs(correlations) > threshold].index.tolist()
    categorical_target = "Churn"

    assoc = associations(df[cat_features], nominal_columns="all", plot=False)
    corr_matrix = assoc["corr"]

    threshold = 0.3
    selected_cat_features = corr_matrix[
        corr_matrix.loc[categorical_target] > threshold
    ].index.tolist()

    del selected_cat_features[-1]
    selected_features = []
    selected_features.extend(selected_num_features)
    selected_features.extend(selected_cat_features)
    print(selected_features)
    return selected_features


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


def genModel(num_features, cat_features, X_train, y_train):
    best_params = {"classifier__C": 1, "classifier__solver": "lbfgs"}
    logreg_model = LogisticRegression(
        C=best_params["classifier__C"],
        solver=best_params["classifier__solver"],
        max_iter=1000,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", logreg_model)]
    )

    pipeline.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipeline, "logreg_model.joblib")
    joblib.dump(pipeline, "model.pkl")


def modelDev(df):
    target = "ChurnTarget"
    selected_features = featureselection(df)
    X = df[selected_features]
    y = df[target]

    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    num_features = X.select_dtypes(include=["number"]).columns.tolist()

    # Splitting data into Train, Validation, and Test Set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # Prepare the preprocessing step
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train the logistic regression model
    pipeline.fit(X_train, y_train)

    # Evaluate on the validation set
    y_val_pred = pipeline.predict(X_val)
    print(
        "Validation Classification Report:\n", classification_report(y_val, y_val_pred)
    )

    # Evaluate on the test set
    y_test_pred = pipeline.predict(X_test)
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

    #####Skipping the step to optimize the model#############
    ### ideally find the best model after parameter tuning and then select that model######
    genModel(num_features, cat_features, X_train, y_train)


def main():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ########DA#######
    # statsInfo(df)
    # charts(df)
    # correlationAnalysis(df)

    #####ML Dev#########
    modelDev(df)


main()

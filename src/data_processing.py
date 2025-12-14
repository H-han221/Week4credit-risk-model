import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data"""
    return pd.read_csv("data/raw/data.csv")


def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level aggregate features
    """
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_amount=("Amount", "std"),
        )
        .reset_index()
    )

    return agg_df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from transaction timestamp
    """
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df


def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Build sklearn preprocessing pipeline
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def prepare_model_data(raw_path: str):
    """
    Full feature engineering workflow
    """
    # Load data
    df = load_data(raw_path)

    # Extract time features
    df = extract_time_features(df)

    # Aggregate customer-level data
    customer_agg = aggregate_customer_features(df)

    # Merge back categorical & time features (using last transaction)
    last_tx = (
        df.sort_values("TransactionStartTime")
        .groupby("CustomerId")
        .last()
        .reset_index()
    )

    final_df = customer_agg.merge(
        last_tx[
            [
                "CustomerId",
                "transaction_hour",
                "transaction_day",
                "transaction_month",
                "transaction_year",
                "ProductCategory",
                "ChannelId",
                "CurrencyCode",
                "CountryCode",
                "ProviderId",
                "PricingStrategy",
            ]
        ],
        on="CustomerId",
        how="left",
    )

    numerical_features = [
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    ]

    categorical_features = [
        "ProductCategory",
        "ChannelId",
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "PricingStrategy",
    ]

    preprocessor = build_preprocessing_pipeline(
        numerical_features, categorical_features
    )

    return final_df, preprocessor

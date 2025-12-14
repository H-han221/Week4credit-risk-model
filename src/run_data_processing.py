from src.data_processing import prepare_model_data

RAW_DATA_PATH = "data/raw/data.csv"

if __name__ == "__main__":
    df, preprocessor = prepare_model_data(RAW_DATA_PATH)

    print("Feature engineering completed")
    print("Shape:", df.shape)


import polars as pl

def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Minimal, clinically-safe preprocessing.
    XGBoost handles NaNs directly.
    """

    # Fix obvious impossible values (example)
    df = df.with_columns([
        pl.when(pl.col("HR") < 0).then(None).otherwise(pl.col("HR")).alias("HR"),
        pl.when(pl.col("Temp") < 25).then(None).otherwise(pl.col("Temp")).alias("Temp"),
    ])

    return df

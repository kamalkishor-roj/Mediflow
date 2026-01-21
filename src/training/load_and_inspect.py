#Load CSV Using Polars
import polars as pl
from pathlib import Path

DATA_PATH = Path("data/raw/sepsis_data.csv")
df = pl.read_csv(
    DATA_PATH,
    infer_schema_length=5000,
    null_values=["", "NA", "NaN", "null"]
)
print("Dataset loaded successfully")

#Basic Sanity Checks
print("Shape:", df.shape)
print("Number of columns:", len(df.columns))
print("First 5 columns:", df.columns[:5])
#Inspect Column Schema
print("\nSchema:")
print(df.schema)
#Preview Sample Rows
print("\nSample data:")
print(df.head(5))


#Count Missing Values Per Column
missing_counts = df.select([
    pl.col(col).null_count().alias(col)
    for col in df.columns
])
print("\nMissing values per column:")
print(missing_counts)

#Missing Percentage per Column
missing_percentage = df.select([
    (pl.col(col).null_count() / df.height * 100)
    .round(2)
    .alias(col)
    for col in df.columns
])
print("\nMissing percentage per column:")
print(missing_percentage)

#convert to Human-Readable Table
missing_table = missing_percentage.transpose(
    include_header=True,
    header_name="feature",
    column_names=["missing_pct"]
)
print("\nMissingness summary:")
print(missing_table.sort("missing_pct", descending=True))

print([col for col in df.columns if "sepsis" in col.lower() or "label" in col.lower()])


df = df.filter(pl.col("SepsisLabel").is_not_null())

#Check Class Balance
print(df["SepsisLabel"].value_counts())

import pandas as pd

# Read Excel file
excel_file = "data/train/Gen_AI_Dataset.xlsx"

# Read Train-Set sheet
train_df = pd.read_excel(excel_file, sheet_name="Train-Set")
print(f"Train-Set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Columns: {list(train_df.columns)}")

# Read Test-Set sheet
test_df = pd.read_excel(excel_file, sheet_name="Test-Set")
print(f"\nTest-Set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"Columns: {list(test_df.columns)}")

# Group train data by query to combine multiple URLs per query
train_grouped = (
    train_df.groupby("Query")["Assessment_url"]
    .apply(lambda x: "|".join(x))
    .reset_index()
)
train_grouped.columns = ["query", "relevant_assessment_urls"]
print(f"\nTrain-Set after grouping: {train_grouped.shape[0]} unique queries")

# Save train.csv
train_grouped.to_csv("data/train/train.csv", index=False)
print(f"Saved: data/train/train.csv")

# Save test.csv
test_df.columns = ["query"] if len(test_df.columns) == 1 else test_df.columns
test_df.to_csv("data/test/test.csv", index=False)
print(f"Saved: data/test/test.csv")

# Show samples
print("\n" + "=" * 60)
print("TRAIN SAMPLE (first 3):")
print("=" * 60)
print(train_grouped.head(3))

print("\n" + "=" * 60)
print("TEST SAMPLE (first 3):")
print("=" * 60)
print(test_df.head(3))

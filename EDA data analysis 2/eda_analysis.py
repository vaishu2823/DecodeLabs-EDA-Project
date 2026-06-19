import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for saving images
os.makedirs("EDA_Images", exist_ok=True)

# Load Excel Dataset
df = pd.read_excel("dataset.xlsx")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Mean
print("\nMean:")
print(df.mean(numeric_only=True))

# Median
print("\nMedian:")
print(df.median(numeric_only=True))

# Correlation Matrix
print("\nCorrelation Matrix:")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("EDA_Images/correlation_heatmap.png")
plt.show()

# Histograms
df.hist(figsize=(12, 8))
plt.suptitle("Histogram of Numerical Columns")
plt.tight_layout()
plt.savefig("EDA_Images/histogram.png")
plt.show()

# Boxplots for Outlier Detection
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"EDA_Images/{col}_boxplot.png")
    plt.show()

# Unique Values
for col in df.columns:
    print(f"\nUnique values in {col}:")
    print(df[col].nunique())

print("\nEDA Completed Successfully!")
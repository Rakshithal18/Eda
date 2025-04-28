import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\raksh\Downloads\train.csv")

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nSurvival Value Counts:")
print(df['Survived'].value_counts())

print("\nPassenger Class Value Counts:")
print(df['Pclass'].value_counts())

# 4. Data Visualization

## a) Pairplot
sns.pairplot(df, hue='Survived', vars=['Pclass', 'Age', 'Fare'])
plt.suptitle('Pairplot of Features Colored by Survival', y=1.02)
plt.show()

## b) Heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

## c) Histograms

# Age Distribution
plt.figure(figsize=(8,5))
df['Age'].hist(bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

## d) Boxplots

# Fare by Passenger Class
plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution Across Passenger Classes')
plt.show()

## e) Scatterplot

# Age vs Fare colored by Survival
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare Colored by Survival')
plt.show()


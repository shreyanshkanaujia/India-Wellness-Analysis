# ============================================================
# PYTHON PROJECT: India Government Wellness Centers Analysis
# Source: Ministry of Health & Family Welfare, Govt. of India
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures  

# ============================================================
# PHASE 1: LOAD & CLEAN DATA
# ============================================================

# Load CSV
df = pd.read_csv("HealthProject.csv")   

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Clean Longitude (had embedded newlines/whitespace)
df['Longitude'] = df['Longitude'].astype(str).str.strip().replace('nan', np.nan)
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')

# Clean DoctorCount
df['DoctorCount'] = pd.to_numeric(df['DoctorCount'], errors='coerce').fillna(0).astype(int)

# Standardise CityName casing
df['CityName'] = df['CityName'].str.strip().str.title()

df.reset_index(drop=True, inplace=True)

print("\n Data Loaded & Cleaned!")
print(f"Shape after cleaning : {df.shape}")
print(f"Total Centers        : {len(df)}")
print(f"Cities covered       : {df['CityName'].nunique()}")
print(f"Medicine categories  : {df['Category'].nunique()}")
print("\nFirst 5 rows:")
print(df[['CityName', 'Category', 'DoctorCount', 'WellnessCenterCode']].head())
print("\nBasic Statistics (DoctorCount):")
print(df['DoctorCount'].describe())


# ============================================================
# PHASE 2: EDA — 5 OBJECTIVES
# ============================================================

# OBJECTIVE 1: Distribution of Wellness Centers by Category
category_dist = df['Category'].value_counts().reset_index()
category_dist.columns = ['Category', 'Center_Count']
category_dist['Percentage'] = (category_dist['Center_Count'] / len(df) * 100).round(2)

print("\n" + "=" * 55)
print("OBJECTIVE 1: Wellness Centers by Medicine Category")
print("=" * 55)
print(category_dist.to_string(index=False))

# OBJECTIVE 2: Top 10 Cities by Number of Wellness Centers
city_centers = df.groupby('CityName').size().reset_index(name='Center_Count')
city_centers = city_centers.sort_values('Center_Count', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 2: Top 10 Cities by Wellness Center Count")
print("=" * 55)
print(city_centers.head(10).to_string(index=False))

# OBJECTIVE 3: Doctor Availability by Category
doctor_by_cat = df.groupby('Category')['DoctorCount'].agg(
    Total_Doctors='sum',
    Avg_Doctors_per_Center='mean',
    Max_Doctors='max'
).round(2).reset_index()
doctor_by_cat = doctor_by_cat.sort_values('Total_Doctors', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 3: Doctor Availability by Category")
print("=" * 55)
print(doctor_by_cat.to_string(index=False))

# OBJECTIVE 4: City-wise Doctor Strength (Top 10)
city_doctors = df.groupby('CityName')['DoctorCount'].sum().reset_index()
city_doctors.columns = ['CityName', 'Total_Doctors']
city_doctors = city_doctors.sort_values('Total_Doctors', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 4: Top 10 Cities by Total Doctor Count")
print("=" * 55)
print(city_doctors.head(10).to_string(index=False))

# OBJECTIVE 5: Category Mix per Top 5 Cities (Cross-tab)
top5_cities = city_centers.head(5)['CityName'].tolist()
cross = df[df['CityName'].isin(top5_cities)].groupby(
    ['CityName', 'Category']).size().unstack(fill_value=0)

print("\n" + "=" * 55)
print("OBJECTIVE 5: Category Mix in Top 5 Cities")
print("=" * 55)
print(cross.to_string())


# ============================================================
# PHASE 3: VISUALISATIONS (Original)
# ============================================================

sns.set_style("whitegrid")

# PLOT 1: Category Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].pie(category_dist['Center_Count'], labels=category_dist['Category'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2", len(category_dist)))
axes[0].set_title("Obj 1A: Centers by Category (Pie)", fontweight='bold')
bars = axes[1].bar(category_dist['Category'], category_dist['Center_Count'], color=sns.color_palette("Set2", len(category_dist)), edgecolor='black', linewidth=0.5)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(int(bar.get_height())), ha='center', fontsize=9, fontweight='bold')
axes[1].set_title("Obj 1B: Centers by Category (Bar)", fontweight='bold')
plt.suptitle("Objective 1: Wellness Center Distribution by Medicine Category", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# PLOT 2: Top 10 Cities
top10 = city_centers.head(10)
plt.figure(figsize=(12, 6))
bars = plt.barh(top10['CityName'][::-1], top10['Center_Count'][::-1], color='steelblue', edgecolor='black', linewidth=0.5)
for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va='center', fontsize=9)
plt.title("Objective 2: Top 10 Cities by Wellness Center Count", fontsize=13, fontweight='bold')
plt.show()

# PLOT 3: Doctor Availability
x = np.arange(len(doctor_by_cat))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width/2, doctor_by_cat['Total_Doctors'], width, label='Total Doctors', color='coral', edgecolor='black')
ax.bar(x + width/2, doctor_by_cat['Avg_Doctors_per_Center']*10, width, label='Avg per Center ×10', color='steelblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(doctor_by_cat['Category'])
ax.set_title("Objective 3: Doctor Availability by Category", fontsize=13, fontweight='bold')
ax.legend()
plt.show()

# PLOT 4: Top 10 Cities by Doctor Count
top10_doc = city_doctors.head(10)
plt.figure(figsize=(12, 6))
plt.bar(top10_doc['CityName'], top10_doc['Total_Doctors'], color=sns.color_palette("coolwarm", len(top10_doc)), edgecolor='black')
plt.title("Objective 4: Top 10 Cities by Total Doctor Count", fontsize=13, fontweight='bold')
plt.xticks(rotation=35, ha='right')
plt.show()

# PLOT 5: Heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
plt.title("Objective 5: Category Mix Heatmap — Top 5 Cities", fontsize=13, fontweight='bold')
plt.show()


# ============================================================
# PHASE 4: ENHANCED MACHINE LEARNING
# ============================================================
print("\n" + "=" * 55)
print("PHASE 4: ENHANCED LINEAR REGRESSION")
print("======================================================")

# 1. New Numeric Features (Parameters)
city_counts = df['CityName'].value_counts().to_dict()
df['City_Density'] = df['CityName'].map(city_counts)

cat_counts = df['Category'].value_counts().to_dict()
df['Cat_Freq'] = df['Category'].map(cat_counts)

df['City_Cat_Interaction'] = df['City_Density'] * df['Cat_Freq']

# 2. Encode Labels
le_cat  = LabelEncoder()
le_city = LabelEncoder()
df['Cat_Enc']  = le_cat.fit_transform(df['Category'])
df['City_Enc'] = le_city.fit_transform(df['CityName'])

# 3. Define X and y
features = ['Cat_Enc', 'City_Enc', 'City_Density', 'Cat_Freq', 'City_Cat_Interaction']
X = df[features]
y = df['DoctorCount']

# 4. Polynomial Transform 
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred_all = model.predict(X_poly)

# Accuracy Metrics
mae  = mean_absolute_error(y, y_pred_all)
r2   = r2_score(y, y_pred_all)

print(f"\n📐 Model Accuracy Metrics:")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")

# PLOT 6: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_all, alpha=0.5, color='steelblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')
plt.title(f"Phase 4: Actual vs Predicted (R² = {r2:.2f})", fontweight='bold')
plt.legend()
plt.show()


# ============================================================
# PHASE 5: BONUS INSIGHTS (Original)
# ============================================================
print("\n" + "=" * 55)
print("PHASE 5: BONUS INSIGHTS")
print("=" * 55)

zero_doc = df[df['DoctorCount'] == 0]
print(f"\n⚠️ Centers with 0 Doctors: {len(zero_doc)}")

ratio = city_doctors.copy().merge(city_centers, on='CityName')
ratio['Doctor_per_Center'] = (ratio['Total_Doctors'] / ratio['Center_Count']).round(2)
print(f"\n📊 Doctor-to-Center Ratio (Top 10 Cities):")
print(ratio[['CityName', 'Total_Doctors', 'Center_Count', 'Doctor_per_Center']].head(10).to_string(index=False))

print("\n✅ PROJECT COMPLETE!")

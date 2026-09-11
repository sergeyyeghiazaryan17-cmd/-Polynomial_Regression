import pandas as pd


df = pd.read_csv('housing.csv')

print(df.head())

import pandas as pd
import numpy as np

# 1. Ստուգում ենք բաց թողնված արժեքները
print("Բաց թողնված արժեքներ (Nulls):")
print(df.isnull().sum())

# Եթե total_bedrooms-ում կան դատարկ արժեքներ, կարող ենք լրացնել միջինով (imputation)
# 2. Առանձնացնում ենք X (հատկանիշներ) և y (թիրախ՝ median_house_value)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

print("\nX-ի չափը:", X.shape)
print("y-ի չափը:", y.shape)
import matplotlib.pyplot as plt
import seaborn as sns

# Միավորում ենք X-ը և y-ը ժամանակավորապես կորելացիան հաշվելու համար
temp_df = X.copy()
temp_df['median_house_value'] = y

# Հաշվում ենք կորելացիան միայն թվային սյունակների միջև
corr_matrix = temp_df.select_dtypes(include=['int64', 'float64']).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Կորելացիայի Մատրից (Correlation Matrix)")
plt.show()



plt.figure(figsize=(10, 7))
plt.scatter(df['longitude'], df['latitude'], c=df['median_house_value'], cmap='jet', alpha=0.4, s=df['population']/100)
plt.colorbar(label='Միջին գին (Median House Value)')
plt.xlabel('Longitude (Երկայնություն)')
plt.ylabel('Latitude (Լայնություն)')
plt.title('Գների բաշխումն ըստ աշխարհագրական դիրքի')
plt.show()



import numpy as np

# Կիրառում ենք log1p թիրախային սյունակի վրա, որպեսզի բաշխումը մոտեցնենք նորմալին
y_transformed = np.log1p(y)

print("Սկզբնական y-ի արժեքներ (առաջին 5-ը):", y.head().values)
print("Փոխակերպված y-ի արժեքներ (առաջին 5-ը):", y_transformed.head().values)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ստեղծում ենք համեմատական գրաֆիկներ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Սկզբնական բաշխում (նկատելի է աջ շեղվածությունը և կտրվածքը 500k-ի վրա)
sns.histplot(y, kde=True, ax=axes[0], color='tomato', bins=50)
axes[0].set_title('Սկզբնական Գների Բաշխում (Original)')
axes[0].set_xlabel('Գին')
axes[0].set_ylabel('Հաճախականություն')

# 2. Փոխակերպված բաշխում (log1p-ից հետո դառնում է ավելի նորմալանման)
y_transformed = np.log1p(y)
sns.histplot(y_transformed, kde=True, ax=axes[1], color='dodgerblue', bins=50)
axes[1].set_title('Լոգարիթմական Փոխակերպում (Log-Transformed)')
axes[1].set_xlabel('Log(Գին + 1)')
axes[1].set_ylabel('Հաճախականություն')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np

# Ֆունկցիա՝ outliers-ների քանակը հաշվելու համար
def count_outliers(data, columns):
    outliers_info = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Հաշվում ենք սահմաններից դուրս ընկնող տողերի քանակը
        count = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].count()
        outliers_info[col] = count
        
    return pd.Series(outliers_info)

# Ստուգում ենք հիմնական թվային սյունակները, որոնք հակված են մեծ շեղումների
numeric_cols = ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
print("Outliers-ների քանակն ըստ սյունակների․")
print(count_outliers(df, numeric_cols))




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Առանձնացնում ենք X-ը և y-ը (հանում ենք ocean_proximity-ն ու թիրախային սյունակը)
X = df.drop(columns=['median_house_value', 'ocean_proximity'])
y = df['median_house_value']

# 2. Բաժանում ենք train և test բազաների
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Ստեղծում ենք ավելի հզոր Pipeline, որը ներառում է Imputer (NaN-երը լրացնելու համար)
poly_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # Լրացնում է բաց թողնված արժեքները միջինով
    ('scaler', StandardScaler()),                    # Մասշտաբավորում է
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # Պոլինոմիալ աստիճաններ
    ('model', LinearRegression())                    # Մոդել
])

# 4. Մարզում ենք մոդելը
poly_pipeline.fit(X_train, y_train)

# 5. Կանխատեսումներ և գնահատում
y_pred_train = poly_pipeline.predict(X_train)
y_pred_test = poly_pipeline.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score:  {test_r2:.4f}")
print(f"Test RMSE:      {test_rmse:.4f}")

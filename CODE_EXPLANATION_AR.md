# شرح تفصيلي للكود سطرًا بسطر

يقدم هذا المستند تفصيلاً دقيقًا لكود Python المستخدم في دفتر الملاحظات `electricity_price_prediction.ipynb`.

## 1. استيراد المكتبات (Import Libraries)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
%matplotlib inline
```

- `import numpy as np`: استيراد مكتبة NumPy، وهي أساسية للحوسبة العلمية في Python، وتستخدم بشكل رئيسي للتعامل مع المصفوفات (Arrays). تم تسميتها اختصارًا `np`.
- `import pandas as pd`: استيراد مكتبة pandas، المستخدمة لمعالجة وتحليل البيانات (إنشاء DataFrames). تم تسميتها اختصارًا `pd`.
- `import matplotlib.pyplot as plt`: استيراد وحدة الرسم من مكتبة Matplotlib لإنشاء رسوم بيانية ثابتة. تم تسميتها اختصارًا `plt`.
- `import seaborn as sns`: استيراد مكتبة Seaborn، وهي مكتبة لتصور البيانات الإحصائية تعتمد على Matplotlib وتوفر واجهة عالية المستوى لرسم رسومات جذابة.
- `from sklearn.linear_model import LinearRegression`: استيراد خوارزمية الانحدار الخطي (Linear Regression) من مكتبة scikit-learn، والتي سيتم استخدامها لبناء نموذج التنبؤ.
- `from sklearn.metrics import mean_squared_error, r2_score`: استيراد دوال القياس لتقييم أداء النموذج (متوسط مربع الخطأ RMSE ومعامل التحديد R-squared).
- `from sklearn.preprocessing import StandardScaler`: استيراد أداة لتوحيد الميزات (Features) عن طريق إزالة المتوسط وتغيير المقياس ليكون التباين واحدًا (Unit Variance).
- `%matplotlib inline`: "أمر سحري" خاص بـ Jupyter Notebook يضمن عرض الرسوم البيانية مباشرة أسفل خلية الكود التي أنتجتها.

## 2. تحميل واستكشاف البيانات (Load and Explore Data)

```python
# Load the datasets
train_data = pd.read_csv("2018_CI_Assignment_Training_Data.csv")
test_data = pd.read_csv("2018_CI_Assignment_Testing_Data.csv")

# Display first few rows
print("Training Data Head:")
display(train_data.head())

# Basic statistics
print("\nTraining Data Description:")
display(train_data.describe())
```

- `train_data = pd.read_csv(...)`: قراءة مجموعة بيانات التدريب من ملف CSV وتخزينها في إطار بيانات pandas يسمى `train_data`.
- `test_data = pd.read_csv(...)`: قراءة مجموعة بيانات الاختبار من ملف CSV وتخزينها في إطار بيانات pandas يسمى `test_data`.
- `print("Training Data Head:")`: طباعة عنوان توضيحي.
- `display(train_data.head())`: عرض أول 5 صفوف من بيانات التدريب. هذا يساعد في فهم هيكل البيانات (الأعمدة مثل `T(t)`, `D(t)`, `P(t+1)`).
- `display(train_data.describe())`: إنشاء إحصائيات وصفية (العدد، المتوسط، الانحراف المعياري، الحد الأدنى، الحد الأقصى، الأرباع) لكل عمود رقمي. هذا ضروري لاكتشاف توزيع البيانات والقيم الشاذة المحتملة.

## 3. تصور البيانات - توزيع الأسعار (Data Visualization - Price Distribution)

```python
plt.figure(figsize=(10, 6))
sns.histplot(train_data.iloc[:, 6], bins=30, kde=True)
plt.title('Distribution of Electricity Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

- `plt.figure(figsize=(10, 6))`: إنشاء شكل جديد للرسم بحجم محدد (عرض 10 بوصات، ارتفاع 6 بوصات).
- `sns.histplot(train_data.iloc[:, 6], bins=30, kde=True)`: استخدام Seaborn لرسم مدرج تكراري (Histogram) للعمود السابع (الفهرس 6)، والذي يقابل المتغير المستهدف `P(t+1)` (السعر).
  - `bins=30`: تقسيم البيانات إلى 30 فترة.
  - `kde=True`: إضافة خط تقدير كثافة النواة (منحنى ناعم) فوق المدرج التكراري لإظهار شكل التوزيع.
- `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`: تعيين العنوان وتسميات المحاور X و Y.
- `plt.show()`: عرض الرسم البياني.

## 4. تصور البيانات - خريطة الارتباط الحرارية (Data Visualization - Correlation Heatmap)

```python
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

- `plt.figure(figsize=(12, 8))`: إعداد حجم الشكل.
- `train_data.corr()`: حساب مصفوفة الارتباط لإطار البيانات، والتي توضح كيفية ارتباط كل متغير بكل متغير آخر (تتراوح القيم من -1 إلى 1).
- `sns.heatmap(...)`: تصور هذه المصفوفة كخريطة حرارية ملونة.
  - `annot=True`: كتابة قيمة الارتباط داخل كل خلية.
  - `cmap='coolwarm'`: تعيين نظام الألوان (أزرق للارتباط السلبي، أحمر للإيجابي).
  - `fmt='.2f'`: تنسيق الأرقام لمنزلتين عشريتين.
- `plt.tight_layout()`: ضبط معلمات المخططات الفرعية تلقائيًا لإعطاء مسافات مناسبة وضمان عدم تداخل التسميات.

## 5. معالجة البيانات - إزالة القيم الشاذة (Data Preprocessing - Outlier Removal)

```python
def remove_outliers(data, col_idx=6):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data.iloc[:, col_idx], 25)
    q3 = np.percentile(data.iloc[:, col_idx], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data.iloc[:, col_idx] >= lower_bound) & (data.iloc[:, col_idx] <= upper_bound)]

# Remove outliers
train_data_clean = remove_outliers(train_data)
print("Original training data size:", len(train_data))
print("Training data size after removing outliers:", len(train_data_clean))
```

- `def remove_outliers(data, col_idx=6):`: تعريف دالة لتصفية القيم المتطرفة. تقوم افتراضيًا بفحص العمود ذو الفهرس 6 (عمود السعر).
- `q1 = np.percentile(...)`: حساب المئين 25 (الربع الأول) للبيانات.
- `q3 = np.percentile(...)`: حساب المئين 75 (الربع الثالث).
- `iqr = q3 - q1`: حساب المدى الربيعي (IQR)، وهو مقياس للتشتت الإحصائي.
- `lower_bound` / `upper_bound`: تحديد النطاق للبيانات "الطبيعية". أي نقطة بيانات أقل من `Q1 - 1.5*IQR` أو أعلى من `Q3 + 1.5*IQR` تعتبر قيمة شاذة.
- `return data[...]`: إرجاع نسخة مصفاة من البيانات تحتوي فقط على الصفوف التي تقع قيمها ضمن الحدود.
- `train_data_clean = ...`: تطبيق هذه الدالة على بيانات التدريب.
- `print(...)`: عرض عدد الصفوف قبل وبعد التنظيف لإظهار عدد القيم الشاذة التي تمت إزالتها.

## 6. تجهيز الميزات والهدف (Prepare Features and Target)

```python
# Prepare features and target
X_train = train_data_clean.iloc[:, :-1]  # All columns except the last one
X_test = test_data.iloc[:, :-1]
y_train = train_data_clean.iloc[:, -1]   # Last column is the target
y_test = test_data.iloc[:, -1]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- `X_train = train_data_clean.iloc[:, :-1]`: اختيار جميع الصفوف وجميع الأعمدة _ما عدا_ العمود الأخير لتكون ميزات الإدخال (درجة الحرارة، الطلب، إلخ).
- `y_train = train_data_clean.iloc[:, -1]`: اختيار جميع الصفوف والعمود الأخير _فقط_ ليكون المتغير المستهدف (السعر).
- يتم إجراء نفس التقسيم لـ `X_test` و `y_test`.
- `scaler = StandardScaler()`: تهيئة كائن التوحيد القياسي (Scaler).
- `X_train_scaled = scaler.fit_transform(X_train)`: ملاءمة الـ Scaler مع بيانات التدريب (حساب المتوسط والانحراف المعياري) ثم تحويل (توحيد مقياس) بيانات التدريب.
- `X_test_scaled = scaler.transform(X_test)`: تحويل بيانات الاختبار باستخدام _نفس_ المتوسط والانحراف المعياري المحسوبين من مجموعة التدريب. هذا يضمن الاتساق.

## 7. تدريب نموذج الانحدار الخطي (Train Linear Regression Model)

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Model training complete")
```

- `model = LinearRegression()`: تهيئة كائن نموذج الانحدار الخطي.
- `model.fit(X_train_scaled, y_train)`: تدريب النموذج باستخدام ميزات التدريب الموحدة (`X`) والقيم المستهدفة (`y`). يتعلم النموذج المعاملات (الأوزان) للمعادلة الخطية.

## 8. تقييم النموذج (Model Evaluation)

```python
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nModel Performance:")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
```

- `model.predict(...)`: استخدام النموذج المدرب لتوليد تنبؤات الأسعار لكل من مجموعتي التدريب والاختبار.
- `mean_squared_error(...)`: حساب متوسط مربع الفرق بين القيم الحقيقية والقيم المتوقعة.
- `np.sqrt(...)`: أخذ الجذر التربيعي لـ MSE للحصول على جذر متوسط مربع الخطأ (RMSE)، والذي يكون بنفس وحدات المتغير المستهدف (السعر).
- `r2_score(...)`: حساب معامل التحديد (R-squared)، الذي يمثل نسبة التباين في المتغير التابع التي يمكن تفسيرها بواسطة النموذج.
- `print(...)`: طباعة المقاييس المحسوبة لتقييم أداء النموذج.

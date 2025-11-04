#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


# In[2]:


np.random.seed(42)

# Number of samples and features
n_samples = 5000
n_features = 3


# In[3]:


X = np.random.randn(n_samples, n_features)


# In[4]:


weights = np.array([1.5, -2.0, 1.0])
bias = 0.5
linear_combination = X @ weights + bias
probability = 1 / (1 + np.exp(-linear_combination))  # Sigmoid for probability
y = (probability > 0.5).astype(int)  # Binary labels


# In[5]:


df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
df['label'] = y

print(df.head())


# In[6]:


import pandas as pd

# Show all rows
pd.set_option('display.max_rows', 5000)

print(df)


# In[7]:


df.to_csv('dummy_dataset.csv', index=False)


# In[8]:


print(df.head(100))  # first 100 rows
print(df.iloc[100:200])  # rows 100 to 199


# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assuming df is your DataFrame created previously
X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['label']

# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[10]:


# 2-way split: 80% train, 20% test
X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2-Way Split:")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# In[11]:


# 3-way split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 ≈ 15% / 85%

print("3-Way Split:")
print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)


# In[12]:


from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("K-Fold Cross Validation (5 folds):")
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"  Train size: {len(train_idx)}")
    print(f"  Test size: {len(test_idx)}")


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the dummy dataset
np.random.seed(42)
n_samples = 5000
n_features = 3

X = np.random.randn(n_samples, n_features)
weights = np.array([1.5, -2.0, 1.0])
bias = 0.5
linear_combination = X @ weights + bias
probability = 1 / (1 + np.exp(-linear_combination))  # Sigmoid
y = (probability > 0.5).astype(int)

df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
df['label'] = y

# Step 2: Split the data into train/test (80/20)
X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on test data: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Create dummy dataset (same as before)
np.random.seed(42)
n_samples = 5000
n_features = 3

X = np.random.randn(n_samples, n_features)
weights = np.array([1.5, -2.0, 1.0])
bias = 0.5
linear_combination = X @ weights + bias
probability = 1 / (1 + np.exp(-linear_combination))  # Sigmoid
y = (probability > 0.5).astype(int)

df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
df['label'] = y

# Step 2: Train/test split
X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)
print("Predicted labels on test set:")
print(y_pred[:10])  # print first 10 predictions

# Step 5: Predict on NEW unseen data (example)
new_samples = pd.DataFrame({
    'feature_1': [0.1, -1.2, 0.5],
    'feature_2': [1.0, 0.3, -0.8],
    'feature_3': [-0.5, 0.2, 1.5]
})

# Predict labels for new samples
new_pred = model.predict(new_samples)
print("\nPredictions for new samples:")
print(new_pred)

# Predict probabilities for new samples
new_proba = model.predict_proba(new_samples)
print("\nPredicted probabilities for new samples:")
print(new_proba)


# In[ ]:





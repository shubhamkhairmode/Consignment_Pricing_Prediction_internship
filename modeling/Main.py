#!/usr/bin/env python
# coding: utf-8

# # Consignment Pricing Prediction

# ![1.jpg](attachment:1.jpg)

# ## Problem Statement :
# The market for logistics analytics is expected to develop at a CAGR of 17.3 percent
# from 2019 to 2024, more than doubling in size. This data demonstrates how logistics
# organizations are understanding the advantages of being able to predict what will
# happen in the future with a decent degree of certainty. Logistics leaders may use this
# data to address supply chain difficulties, cut costs, and enhance service levels all at the
# same time.
# 
# The main goal is to predict the consignment pricing based on the available factors in the
# dataset.
# In this project, our goal is to predict the mode of transport for a given shipment using supply chain data. To accomplish this, we will harness the power of TensorFlow and Keras, utilizing a neural network model.
# 
# By analyzing various factors and features within the supply chain data, we aim to develop a reliable model that can accurately predict the appropriate mode of transport for shipments. This predictive capability can assist in optimizing logistics and decision-making processes within the supply chain industry.
# 
# With the TensorFlow/Keras framework, we will construct and train a neural network that learns from the provided data to make predictions. By leveraging advanced techniques such as dense layers and softmax activation, our model will be able to classify and predict the most suitable mode of transport for a given shipment scenario.
# 
# Through this project, we aim to enhance supply chain management by enabling efficient and accurate decision-making in the shipment transportation process. Let's dive into the world of supply chain data and embark on this exciting journey of prediction and optimization!
# 
# **dataset** : https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data

# In[105]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ## Business Understanding / Data Understanding

# In[106]:


# reading dataset
df = pd.read_csv('SCMS_Delivery_History_Dataset.csv')
df.head()


# In[107]:


#check shape
print(f"Total No of Rows: {df.shape[0]} and Columns: {df.shape[1]}")


# In[108]:


# basic info
df.info()


# In[109]:


# total no of unique values in each columns
df.nunique()


# In[110]:


# checking for null values
df.isnull().sum()


# In[111]:


#null values in %ages
df.isnull().mean()*100


# ## EDA

# In[112]:


# heatmap for null values
sns.heatmap(df.isnull())
plt.show()


# In[113]:


# chexking duplicated values
df.duplicated().sum()


# In[114]:


# statasic analysis
df.describe().T


# In[115]:


df.describe(include=object).T


# In[21]:


#loading necessary packages for the dataprep
from dataprep.eda import *
from dataprep.datasets import load_dataset
from dataprep.eda import plot, plot_correlation, plot_missing, plot_diff, create_report


# In[24]:


create_report(df)


# ## Processing

# In[116]:


def preprocess_inputs(df, label_mapping):
    df = df.copy()
    
    # Dropping ID Column Because That Is Unique For Each Shipment & Doesn't Serve Any Process
    df = df.drop('ID', axis=1)
    
    # Dropping Missing Values in Target Rows Of [Shipment Mode] Column as that is our target column and we don't to do prediction on fabricated dat
    
    missing_target_rows = df[df['Shipment Mode'].isna()].index
    df = df.drop(missing_target_rows, axis=0).reset_index(drop=True)
    
    # Filling Missing values
    df['Dosage'] = df['Dosage'].fillna(df['Dosage'].mode()[0])
    df['Line Item Insurance (USD)'] = df['Line Item Insurance (USD)'].fillna(df['Line Item Insurance (USD)'].mean())
    
    # Drop Date Column With Too Many Missing Values
    df = df.drop(['PQ First Sent to Client Date','PO Sent to Vendor Date'],axis=1)
    
    # Extract Date Features
    for column in ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']:
        df[column] = pd.to_datetime(df[column])
        df[column + ' Year'] = df[column].apply(lambda x : x.year)
        df[column + ' Month'] = df[column].apply(lambda x : x.month)
        df[column + ' Day'] = df[column].apply(lambda x : x.day)
        df = df.drop(column, axis=1)
        
    # Drop Numeric Columns for too many missing values
    df = df.drop(['Weight (Kilograms)','Freight Cost (USD)'], axis=1)
    
    # Drop High Cardinality Columns
    df = df.drop(['PQ #','PO / SO #','ASN/DN #'], axis=1)
    
    # Binary Encoding
    df['Fulfill Via'] = df['Fulfill Via'].replace({'Direct Drop':0,'From RDC':1})
    df['First Line Designation'] = df[ 'First Line Designation'].replace({'No':0, 'Yes':1})
    
    # One-Hot Encoding
    for column in df.select_dtypes('object').columns.drop('Shipment Mode'):
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(column, axis=1)
    
    # Splitting The DataFrame into X and y
    y = df['Shipment Mode']
    X = df.drop('Shipment Mode', axis=1)
    
    # Encoding The Labels In Shipment Mode
    y = y.replace(label_mapping)
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test


# In[118]:


LABEL_MAPPING = {
    'Air': 0,
    'Truck': 1,
    'Air Charter': 2,
    'Ocean': 3
}

X_train, X_test, y_train, y_test = preprocess_inputs(df, label_mapping=LABEL_MAPPING)


# In[120]:


X_train.head()


# In[121]:


X_train.shape, X_test.shape


# In[126]:


import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report


# In[128]:


inputs = tf.keras.Input(shape=(771,))
x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.5)(x)  # Add dropout for regularization
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Add dropout for regularization
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Add dropout for regularization
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjust learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Increase patience to allow longer training
            restore_best_weights=True
        )
    ]
)


# In[129]:


y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred, labels=list(LABEL_MAPPING.values()))
clr = classification_report(y_test, y_pred, labels=list(LABEL_MAPPING.values()), target_names=list(LABEL_MAPPING.keys()))

print("Test Set Accuracy: {:.2f}%".format(model.evaluate(X_test, y_test, verbose=0)[1] * 100))


# In[130]:


plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=list(LABEL_MAPPING.keys()))
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=list(LABEL_MAPPING.keys()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)


# #### Credit : shubham khairmode
# #### feedback : shubhambkhairmode@gmail.com

# In[ ]:





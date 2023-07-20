#!/usr/bin/env python
# coding: utf-8

# ## Consignment Pricing Prediction

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

# In[3]:


import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import os
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20,.2f}'.format


# ## Import CSV file

# In[5]:


DataSet = pd.read_csv('SCMS_Delivery_History_Dataset.csv').fillna(0)


# ## Check Total Records in CSV file

# In[6]:


TotalRowCount = len(DataSet)
print("Total Number of Data Count :", TotalRowCount)


# ## Check DataType of CSV file

# In[7]:


DataSet.dtypes


# ## Print first 10 and last 10 recods from DataSet

# In[8]:


DataSet.head(10)


# In[9]:


DataSet.tail(10)


# ## Total 10 Country wise count with graph

# In[13]:


# Drop rows with missing values
DataSet = DataSet.dropna()

# Get the counts of each country and select the top 10
ItemCount = DataSet["Country"].value_counts().nlargest(10)

# Print the top 10 countries with their counts
print("Top 10 Countries Wise Count\n")
print(ItemCount)

# Set the plot context and font scale for Seaborn
sn.set_context("talk", font_scale=1)

# Convert the index of ItemCount to a list and use it as the order for the countplot
top_countries_order = ItemCount.index.tolist()

# Create a bar plot of the top 10 countries
plt.figure(figsize=(22, 6))
sn.countplot(data=DataSet, x='Country', order=top_countries_order)

# Add a title and labels to the plot
plt.title('Top 10 Countries Wise Count\n')
plt.ylabel('Total Count')
plt.xlabel('Country Name')

# Show the plot
plt.show()


# ## Total Pack Price for Top 15 Countries with graph

# In[16]:


TotalPrice = DataSet.groupby(['Country'])['Pack Price'].sum().nlargest(15)
print("Total Pack Price for Top 15 Countries\n")
print(TotalPrice)
plt.figure(figsize=(22,6))
colors = sn.color_palette('viridis', len(TotalPrice))
GraphData=DataSet.groupby(['Country'])['Pack Price'].sum().nlargest(15)
GraphData.plot(kind='bar', color=colors)
plt.ylabel('Total Pack Price')
plt.xlabel('Country Name')


# ## First Line Designation Wise Count

# In[20]:


sn.set_context("talk", font_scale=1)

# Create a count plot of the 'First Line Designation' column
plt.figure(figsize=(6, 4))  # Adjusted the figure size for better visualization

# Use the 'order' parameter to order the bars based on their counts
sn.countplot(data=DataSet, x='First Line Designation',
             order=DataSet['First Line Designation'].value_counts().nlargest(10).index)

plt.title('First Line Designation Wise Count \n')
plt.ylabel('Total Count')
plt.xlabel('First Line Designation')

# Show the plot
plt.show()


# ## Shipment Mode percentage wise Pie Char

# In[21]:


ShippingMode = DataSet["Shipment Mode"].value_counts()
labels = (np.array(ShippingMode.index))
sizes = (np.array((ShippingMode / ShippingMode.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="Shipment Mode")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.iplot(fig, filename="Shipment Mode")


# ## Unquie Manufacturing Site Names

# In[22]:


UniqueItem = DataSet['Manufacturing Site'].unique()
print("All Unique Manufacturing Site \n")
print(UniqueItem)


# ## Shipment Mode, Min and Mean value for Air

# In[23]:


ItemData=DataSet[DataSet['Shipment Mode']=='Air']
print ("The Max Air Shipment Mode is :",ItemData['Unit of Measure (Per Pack)'].max())
print ("The Min Air Shipment is :",ItemData['Unit of Measure (Per Pack)'].min())
ItemTypeMean = ItemData['Unit of Measure (Per Pack)'].mean()
print ("The Mean Air Shipment is :", round(ItemTypeMean,2))


# ## Top 10 Manufacturing Site for all Shipment Mode with Graph

# In[25]:


sn.set_context("talk", font_scale=1)
plt.figure(figsize=(22, 6))
TopFiveManufacturingSite = DataSet.groupby('Manufacturing Site').size().nlargest(10)
print(TopFiveManufacturingSite)
colors = sn.color_palette('magma', len(TopFiveManufacturingSite))
TopFiveManufacturingSite.plot(kind='bar', color=colors)

plt.title('Top 10 Manufacturing Site \n')
plt.ylabel('Total Count')
plt.xlabel('Manufacturing Site Name')

# Show the plot
plt.show()


# ## Top 10 Manufacturing Site for Air Shipment Mode with Graph

# In[26]:


# Top 10 Air Shipment Mode in Bar Chart
ItemData=DataSet[DataSet['Shipment Mode']=='Air']
DataSet[DataSet["Shipment Mode"]=='Air']['Manufacturing Site'].value_counts()[0:10].to_frame().plot.bar(figsize=(22,6))
ItemSupplier = DataSet[DataSet["Shipment Mode"]=='Air']['Manufacturing Site'].value_counts()[0:10]
print("Top 10 Air Manufacturing Site \n")
print(ItemSupplier)
plt.title('Top 10 Air Manufacturing Site\n')
plt.ylabel('Air Count')
plt.xlabel('Manufacturing Site')


# ## Shipment Mode and Pack Price in Bar Plot Graph

# In[28]:


plt.subplots(figsize=(18, 6))
plt.xticks(rotation=90)
sn.barplot(x=DataSet['Shipment Mode'], y=DataSet['Pack Price'])

plt.title('Pack Price vs. Shipment Mode\n')
plt.xlabel('Shipment Mode')
plt.ylabel('Pack Price')

# Show the plot
plt.show()


# ## Conclusion
# - Top Country for Pack Price : Nigeria - 25,620.72
# - Top Shipping Mode : Air
# - The Max Air Shipment Mode is : 1000
# - The Min Air Shipment is : 1
# - The Mean Air Shipment is : 82.35
# - Top Manufacturing Site : Aurobindo Unit III, India - 3172
# - Top Air Manufacturing Site : Aurobindo Unit III, India - 1694

# #### Credit : shubham khairmode
# #### feedback : shubhambkhairmode@gmail.com

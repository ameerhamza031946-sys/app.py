import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.title("Cancer Data Dashboard")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('Cancer_Data.csv')
    return data

data = load_data()

# Display data overview
st.header("Data Overview")
st.write("First 5 rows of the dataset:")
st.dataframe(data.head())

# Diagnosis distribution
st.header("Diagnosis Distribution")
diagnosis_counts = data['diagnosis'].value_counts()
st.write("Count of Malignant (M) and Benign (B) cases:")
st.write(diagnosis_counts)

# Pie chart
fig, ax = plt.subplots()
ax.pie(diagnosis_counts, labels=diagnosis_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Histograms for mean features
st.header("Feature Distributions")
mean_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

for feature in mean_features:
    st.subheader(f"Distribution of {feature}")
    fig, ax = plt.subplots()
    sns.histplot(data[feature], kde=True, ax=ax)
    st.pyplot(fig)

# Correlation heatmap
st.header("Correlation Heatmap")
corr = data.select_dtypes(include=['float64', 'int64']).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

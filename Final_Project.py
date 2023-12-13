#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Tatyana Guzman
# ### 12/07/2023

# #### 1) Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[6]:
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


s = pd.read_csv("social_media_usage.csv")
print("Dimensions of the dataframe:", s.shape)


# ***

# #### 2) Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[7]:


def clean_sm(x):
    return np.where(x == 1, 1, 0)

toy_df = pd.DataFrame({'col1': [1, 0, 1], 'col2': [0, 1, 1]})
toy_df.apply(clean_sm)


# #### 3) Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[8]:


ss = s.copy() 
ss['sm_li'] = clean_sm(ss['web1h'])

ss = ss[ss['age'] <= 98]
ss = ss[ss['income'] <= 9]
ss = ss[ss['educ2'] <= 8]
ss['female'] = np.where(ss['gender'] == 2, 0, 1)
ss['parent'] = np.where(ss['par'] == 2, 0, 1)
ss['married'] = np.where(ss['marital'] >= 2, 0, 1)
ss.dropna(inplace=True)
ss = ss[['income', 'educ2', 'parent','married', 'age', 'female', 'sm_li']]

X = ss[['income', 'educ2', 'parent','married', 'age', 'female']]
X = sm.add_constant(X)  # Add a constant term to the independent variables
y = ss['sm_li']

# Fit logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Display regression results
print(result.summary())


# #### 4) Create a target vector (y) and feature set (X) 5) Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[10]:


X = ss[['income', 'educ2', 'parent', 'married', 'female', 'age']]
y = ss['sm_li']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### 6) Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[21]:


logreg_model = LogisticRegression(class_weight='balanced', random_state=42)
X_train_np = X_train.to_numpy()
feature_names = X.columns.tolist()
logreg_model.fit(X_train_np, y_train)
logreg_model.feature_names_in_ = feature_names


# #### 7) Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[22]:


y_pred = logreg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2%}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# #### 8) Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[23]:


conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

conf_matrix_df.columns.name = 'Predicted'
conf_matrix_df.index.name = 'Actual'


print("Confusion Matrix:")
print(conf_matrix_df)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.show()


# #### 9) Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[24]:




# Calculate precision, recall, and F1 score by hand
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')

# Use sklearn's classification_report to compare
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# Use the model to make predictions for specific scenarios
scenario_1 = [[8, 7, 0, 1, 1, 42]]  # High income, high education, non-parent, married, female, 42 years old
scenario_2 = [[8, 7, 0, 1, 1, 82]]  # High income, high education, non-parent, married, female, 82 years old

# Predict probabilities for each scenario
prob_scenario_1 = logreg_model.predict_proba(scenario_1)[:, 1]
prob_scenario_2 = logreg_model.predict_proba(scenario_2)[:, 1]

print(f"Probability of LinkedIn usage for Scenario 1: {prob_scenario_1[0]:.4f}")
print(f"Probability of LinkedIn usage for Scenario 2: {prob_scenario_2[0]:.4f}")


# #### 10) Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[25]:


# Define the scenarios
scenario_1 = [[8, 7, 0, 1, 1, 42]]  # High income, high education, non-parent, married, female, 42 years old
scenario_2 = [[8, 7, 0, 1, 1, 82]]  # High income, high education, non-parent, married, female, 82 years old

# Predict probabilities for each scenario
prob_scenario_1 = logreg_model.predict_proba(scenario_1)[:, 1]
prob_scenario_2 = logreg_model.predict_proba(scenario_2)[:, 1]

print(f"Probability of LinkedIn usage for Scenario 1: {prob_scenario_1[0]:.4f}")
print(f"Probability of LinkedIn usage for Scenario 2: {prob_scenario_2[0]:.4f}")


# In[ ]:
# Load the trained model
logreg_model = LogisticRegression(class_weight='balanced', random_state=42)  # Include your trained model instantiation code here

if not hasattr(logreg_model, "feature_names_in_"):
    X_train_np = X_train.to_numpy()
    feature_names = X.columns.tolist()
    logreg_model.fit(X_train_np, y_train)
    logreg_model.feature_names_in_ = feature_names

# Streamlit App
def main(logreg_model):
    st.title("LinkedIn User Prediction App")

    # User Input Form
    st.sidebar.header("User Input Features")
    income = st.sidebar.slider("Income", 1, 10, 5)
    education = st.sidebar.slider("Education", 1, 10, 5)
    parent = st.sidebar.radio("Parent", ["Yes", "No"])
    married = st.sidebar.radio("Married", ["Yes", "No"])
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 98, 30)

    # Convert user input to model-compatible format
    gender_code = 1 if gender == "Female" else 0
    parent_code = 1 if parent == "Yes" else 0
    married_code = 1 if married == "Yes" else 0

    # Make prediction
    prediction_proba = logreg_model.predict_proba([[income, education, parent_code, married_code, gender_code, age]])[:, 1]
    prediction = "LinkedIn User" if prediction_proba >= 0.5 else "Non-LinkedIn User"

    # Display Results
    st.subheader("Prediction Results")
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability of LinkedIn Usage: {prediction_proba[0]:.4f}")

if __name__ == "__main__":
    main(logreg_model)


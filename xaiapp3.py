#imports de bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
from shap import Explainer, Explanation
import streamlit as st
def chamar_dados(path, df_name):
    # Read JSON file
    X = pd.read_json(path)
    
    # Create DataFrame
    df = pd.DataFrame(X)
    
    # Transpose DataFrame
    df = df.transpose()
    
    # Assign the DataFrame to the specified name
    globals()[df_name] = df
    
    return df
dfY=chamar_dados("/Users/anafi/Desktop/data_extraction_internship/y_train_reg.json", df_name='dfY')
df_super_completa_bin = pd.read_csv('C:/Users/anafi/Desktop/data_extraction_internship/Final ML + XAI/df_super_completa_bin.csv')
df_super_completa_df = pd.read_csv('C:/Users/anafi/Desktop/data_extraction_internship/Final ML + XAI/df_super_completa_df.csv')
consumiveis = dfY.columns
features = df_super_completa_bin.drop(columns=dfY.columns)
# Function to filter columns
def filter_columns(df_super_completa_df):
    # Calculate the sum of non-zero values across rows
    non_zero_counts = df_super_completa_df.astype(bool).sum(axis=0)
    
    # Keep columns that start with 'C000' or 'F000' and have more than 15 non-zero values
    columns_to_keep = non_zero_counts[non_zero_counts.index.str.startswith('C000') | non_zero_counts.index.str.startswith('F000')]
    columns_to_keep = columns_to_keep[columns_to_keep > 15].index.tolist()
    
    # Include columns that do not start with 'C000' or 'F000'
    other_columns_to_keep = [col for col in df_super_completa_df.columns if not col.startswith('C000') and not col.startswith('F000')]
    
    # Combine both lists of columns to keep
    columns_to_keep.extend(other_columns_to_keep)
    
    return columns_to_keep

# Get the list of columns to keep
columns_to_keep = filter_columns(df_super_completa_df)

# Filter the DataFrame
df_super_completa_df = df_super_completa_df[columns_to_keep]
# Function to filter columns
def filter_columns(df_super_completa_df):
    # Calculate the sum of non-zero values across rows
    non_zero_counts = df_super_completa_bin.astype(bool).sum(axis=0)
    
    # Keep columns that start with 'C000' or 'F000' and have more than 15 non-zero values
    columns_to_keep = non_zero_counts[non_zero_counts.index.str.startswith('C000') | non_zero_counts.index.str.startswith('F000')]
    columns_to_keep = columns_to_keep[columns_to_keep > 15].index.tolist()
    
    # Include columns that do not start with 'C000' or 'F000'
    other_columns_to_keep = [col for col in df_super_completa_bin.columns if not col.startswith('C000') and not col.startswith('F000')]
    
    # Combine both lists of columns to keep
    columns_to_keep.extend(other_columns_to_keep)
    
    return columns_to_keep

# Get the list of columns to keep
columns_to_keep = filter_columns(df_super_completa_bin)

# Filter the DataFrame
df_super_completa_bin = df_super_completa_bin[columns_to_keep]
st.title('Consumable Predictions Explanations')
# Episode selection
episode_index = st.selectbox('Choose an episode', df_super_completa_bin.index)

# Target column selection
target_columns = [col for col in df_super_completa_df.columns if col.startswith('C000') or col.startswith('F000')]
target_column = st.selectbox('Choose a Consumable', target_columns)

# Analysis type selection
analysis_type = st.radio('Choose analysis type', ['usage', 'quantity'])

def split_data(X, y, test_size1=0.3, test_size2=0.5, random_state=42):
    """
    Split the data into train, validation, and test sets.
    
    Parameters:
    X : pandas DataFrame
        The features dataframe.
    y : pandas DataFrame
        The target dataframe.
    test_size1 : float, optional (default=0.3)
        The proportion of the dataset to include in the first split (train-test).
    test_size2 : float, optional (default=0.5)
        The proportion of the dataset to include in the second split (validation-test).
    random_state : int, optional (default=42)
        Random state for reproducibility.
        
    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test : pandas DataFrame
        The train, validation, and test sets for features and target.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size1, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size2/(1-test_size1), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
def logistic_filter(X_train, X_val, X_test, y_train):
    
    # Initialize and train the logistic regression model
    model = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Standardizing features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Predict probabilities on test and validation sets
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    # Apply decision threshold
    test_predictions = (test_probs > 0.33).astype(int)
    val_predictions = (val_probs > 0.33).astype(int)

    # Get all indices for test and validation sets
    test_indices = list(range(len(X_test)))
    val_indices = list(range(len(X_val)))

    # Adjust validation indices to account for their position after the test set
    val_indices_adjusted = [i + len(X_test) for i in val_indices]

    # Combine test and validation indices
    combined_indices = test_indices + val_indices_adjusted

    return combined_indices
# Plot SHAP global function
# Plot SHAP local function
def plot_shap_local(explainer, shap_values, features, corrected_shap_values, corrected_features, instance_index=episode_index):
    
    # Get the expected value (model's output expected value for no features)
    expected_value = explainer.expected_value

    # If it's binary classification and explainer returns a list, take the correct expected value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # For the positive class

    # Create the SHAP Explanation object
    expl = shap.Explanation(values=shap_values,
                            base_values=expected_value,  # or explainer.expected_value if not binary classification
                            data=features,              # the feature values for the instances we're explaining
                            feature_names=features.columns.tolist())

    # Create a SHAP waterfall plot for the first instance in the test set
    shap.plots.waterfall(expl[instance_index], max_display=10)

    # Show plot
    plt.show()
# Function to plot SHAP global summary and return top features
def plot_shap_global_dt(shap_values, features):
    # Generate SHAP summary plot
    shap.summary_plot(shap_values, features, plot_type="bar")

    # Initialize SHAP JS visualization (assuming it's needed)
    shap.initjs()

    # Calculate feature importance and select top features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_indices = np.argsort(feature_importance)[::-1][:20]
    top_features = features.columns[top_features_indices]

    # Correctly index shap_values and features for plotting
    corrected_shap_values = shap_values[:, top_features_indices]
    corrected_features = features.iloc[:, top_features_indices]

    try:
        corrected_shap_values = corrected_shap_values.astype(float)
    except ValueError as e:
        print("Conversion error:", e)
        # Find and print problematic values
        for row in corrected_shap_values:
            for value in row:
                try:
                    float(value)
                except ValueError:
                    print("Non-convertible value:", value)

    try:
        shap.summary_plot(corrected_shap_values, corrected_features)
    except Exception as e:
        print("Error in SHAP plot:", e)

    return corrected_shap_values, corrected_features, top_features, top_features_indices
# Train and explain the model based on analysis type
if analysis_type == 'usage':
    # Logistic Regression model
    features = df_super_completa_bin.columns.drop(target_columns)
    X = df_super_completa_bin[features]

    # Target values
    y = df_super_completa_bin.loc[:, target_column]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    logistic_reg = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=42)
    logistic_reg.fit(X_train, y_train)

    coefficients = logistic_reg.coef_[0]
    feature_names = X_train.columns# Create a DataFrame with coefficients and feature names
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Calculate the absolute values of the coefficients
    coef_df['AbsCoefficient'] = np.abs(coef_df['Coefficient'])

    # Sort the DataFrame based on the absolute values of the coefficients
    sorted_coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)

    # Select the top 10 and bottom 5 coefficients
    top_10_coef = sorted_coef_df.head(15)

    # Extract the values of the selected features for the given row
    row_values = X_train.iloc[episode_index]
    row_feature_values = row_values[top_10_coef['Feature']]

    st.header(f'Explanation for how {target_column} is predicted in the {episode_index}th episode (Logistic Regression Coefficients)')

    # Predict the class for the given row
    row_interest = X_train.iloc[episode_index, :].values.reshape(1, -1)
    predicted_class = logistic_reg.predict(row_interest)
    st.write(f"The predicted class for {target_column} is: {predicted_class[0]}")

    # Plotting
    # Plotting the coefficients
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.barh(top_10_coef['Feature'], top_10_coef['Coefficient'], color=['blue' if x > 0 else 'red' for x in top_10_coef['Coefficient']])
    plt.xlabel('Coefficient Value')
    plt.title('Top 10 and Bottom 5 Coefficients of Logistic Regression Model')
    plt.gca().invert_yaxis()

    # Plotting the feature values for the given row
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.barh(top_10_coef['Feature'], row_feature_values)
    plt.xlabel('Feature Value')
    plt.title(f'Feature Values for Row {episode_index}')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    

elif analysis_type == 'quantity':
    # Decision Tree model
    features = df_super_completa_bin.columns.drop(target_columns)
    X = df_super_completa_df[features]

    # Target values
    y = df_super_completa_df.loc[:, target_column]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    decision_tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=4, min_samples_split=2, random_state=42)
    decision_tree.fit(X_train, y_train)

    # Predict the class for the given row
    row_interest = X_train.iloc[episode_index, :].values.reshape(1, -1)
    predicted_class = decision_tree.predict(row_interest)

    # Print predicted class
    st.write(f"The predicted amount for {target_column} in the {episode_index}th episode is: {predicted_class[0]}")

    explainer = shap.TreeExplainer(decision_tree, model_output='raw', feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X, check_additivity=False)

    st.header(f'Explanation for how {target_column} is predicted (SHAP Values - Decision Tree)')
    corrected_shap_values, corrected_features, top_features, top_features_indices = plot_shap_global_dt(shap_values, X)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    row_idx = episode_index  # Replace with the desired row index
    specific_row = X.iloc[row_idx, top_features_indices]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.barh(top_features, specific_row)
    plt.xlabel('Feature Value')
    plt.title(f'Feature Values for Episode {row_idx}')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
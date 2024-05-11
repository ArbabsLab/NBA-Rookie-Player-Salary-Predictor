import pandas as pd
import numpy as np 

df = pd.read_csv("Final/archive/nba_2022-23_all_stats_with_salary.csv")
rookie_df = pd.read_csv("Final/archive/nba_rookies_2022-2023.csv")


columns_to_fill = ['FT%', '3P%', '2P%', 'eFG%', 'FG%',  '3PAr', 'FTr', 'TOV%', 'TS%']
df[columns_to_fill] = df[columns_to_fill].fillna(0)

X = df[['Player Name', 'Age','GP', 'GS', 'MP',
       'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
       'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
       'PF', 'PTS', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%',
       'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS',
       'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']]

Y = df['Salary']


from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.linear_model import LassoCV


# Calculate the correlation matrix
correlation_matrix = X.drop(['Player Name'], axis=1).corr()


corr_threshold = 0.8 
correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            # Keep one feature and add the other to the set of correlated features to be dropped
            if colname_i not in correlated_features:
                correlated_features.add(colname_j)


# Drop the correlated features
X_filtered = X.drop(columns=correlated_features)
X_filtered.drop(['Player Name'], axis=1, inplace=True)

print(X_filtered)

# LASSO Regression for additional feature selection
lasso = LassoCV()
lasso.fit(X_filtered, Y)

# Use SelectFromModel to get selected features based on LASSO coefficients
sfm = SelectFromModel(lasso, prefit=True)
selected_features_lasso = X_filtered.columns[sfm.get_support()]

# Convert to a DataFrame
selected_features_df = pd.DataFrame(list(selected_features_lasso), columns=['Selected_Features'])

print(selected_features_df)

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# Train-Test Split
train_df, test_df, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

test_df = pd.concat([train_df, test_df])
y_test = pd.concat([y_train, y_test])



X_train = train_df[selected_features_lasso]
X_test = test_df[selected_features_lasso]



# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Define and Train Regression Models
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree Regressor', DecisionTreeRegressor(random_state=42)),
    ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42))
]

for model_name, model in models:
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate MSE and R2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f'Model: {model_name}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R-squared (R2): {r2:.4f}')
    print('---')


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain the GB Regressor on the full training data
rf_model =  GradientBoostingRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

# Calculate MSE and R2 on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Get predictions, actual values, and player names for the test set into a dataframe
predictions_df = pd.DataFrame({
    'Player Name': test_df['Player Name'],
    'Actual Salary': y_test,
    'Predicted Salary': y_pred
})

# Plot actual vs. predicted salaries
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Salaries on Test Dataset')
plt.xlabel('Actual Salary (USD)')
plt.ylabel('Predicted Salary (USD)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='-', color='red')


plt.show()

# Print MSE and R2
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2): {r2:.4f}')

# Get feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to associate feature names with their importance scores
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create an eye-catching horizontal bar chart
colors = ['#007acc', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color=colors)
plt.title('Top 10 Feature Importances in Predicting NBA Player Salaries', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important features at the top

# Add a cool background
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

plt.show()

predictions_df['Absolute Difference'] = predictions_df['Actual Salary'] - predictions_df['Predicted Salary']
predictions_df['Percentage Difference'] = (predictions_df['Absolute Difference'] / predictions_df['Predicted Salary']) * 100

#Filter for rookies
predictions_df = pd.merge(predictions_df, rookie_df, left_on='Player Name', right_on='Player', how='inner')
#Filter out rookies who played less than 25 games
predictions_df = predictions_df[predictions_df['g'] >= 20]
#Make df more concise
predictions_df = predictions_df[['Player Name', 'Actual Salary', 'Predicted Salary', 'Absolute Difference', 'Percentage Difference']]
print('Rookie Predictions', predictions_df)

print("\nOverpaid Rookie Players:\n", predictions_df.sort_values(by='Percentage Difference', ascending=False).head(5))

print("\nUnderpaid Rookie Players:\n", predictions_df.sort_values(by='Percentage Difference', ascending=True).head(5))




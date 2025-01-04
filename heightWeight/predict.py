# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('/Users/abhaysingh/Documents/Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main/3-Complete Linear Regression/Practicals/height-weight.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Step 2: Visualizing the relationship between 'Weight' and 'Height'
x = df[['Weight']]  # Independent variable (features) - Note: This is in the form of a DataFrame.
y = df['Height']    # Dependent variable (target) - This is a Series

# Scatter plot to visualize the data (commented out)
# plt.scatter(x, y)
# plt.title('Height Prediction vs Weight')
# plt.xlabel('Weight')
# plt.ylabel('Height')
# plt.show()  # Display the plot

# Step 3: Check the correlation between features (this helps us understand if there is a linear relationship)
print(df.corr())

# Step 4: Visualizing the dataset with pairplot (helps to visually explore relationships between multiple features)
# sns.pairplot(df)
# plt.show()  # Show the pairplot to explore the relationship visually

# Step 5: Split the data into training and testing sets
# Train-Test Split: We will use 75% of the data for training and 25% for testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Step 6: Data Preprocessing - Standardizing the data
# Standardization (Z-score scaling) ensures that the features have a mean of 0 and a standard deviation of 1
# This is useful for some models, but in linear regression, it's not strictly necessary unless we have multiple features.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Fit and transform on training data
x_test = scaler.transform(x_test)        # Transform the test data using the same scaler

# Step 7: Apply Simple Linear Regression Model
# Linear Regression is used to model the relationship between 'Weight' and 'Height'
regression = LinearRegression()

# Fit the model on the training data
regression.fit(x_train, y_train)

# Output the model's coefficient (slope) and intercept (y-intercept)
print("Coefficient (slope) of the model:", regression.coef_)
print("Intercept (y-intercept) of the model:", regression.intercept_)

# Step 8: Visualize the best fit line on the training data (commented out)
# plt.scatter(x_train, y_train, color='blue', label='Training Data')
# plt.plot(x_train, regression.predict(x_train), color='red', label='Regression Line')
# plt.title('Best Fit Line - Training Data')
# plt.xlabel('Weight')
# plt.ylabel('Height')
# plt.legend()
# plt.show()  # Show the plot

# Step 9: Predicting the Height using the trained model on test data
y_pred = regression.predict(x_test)

# Step 10: Evaluate the performance of the model using common regression metrics

# Mean Squared Error (MSE): Measures the average squared difference between the actual and predicted values.
mse = mean_squared_error(y_test, y_pred)

# Mean Absolute Error (MAE): Measures the average absolute difference between the actual and predicted values.
mae = mean_absolute_error(y_test, y_pred)

# Root Mean Squared Error (RMSE): The square root of the Mean Squared Error, providing a more interpretable metric in terms of units.
rmse = np.sqrt(mse)

# Display the performance metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Step 11: R-squared (R²) - Coefficient of Determination
# R² indicates the proportion of the variance in the dependent variable (Height) that is predictable from the independent variable (Weight).
r2 = r2_score(y_test, y_pred)

# Display the R² value
print("R-squared (R²):", r2)

# Step 12: Adjusted R-squared (Adjusted R²)
# Adjusted R² takes into account the number of features in the model and adjusts R² accordingly. It helps compare models with different numbers of predictors.
# Formula: Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
# Where:
# n = number of data points
# k = number of predictor variables (features)
n = len(y_test)  # Number of observations in the test set
k = x_test.shape[1]  # Number of predictor variables (in this case, only 'Weight')

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Display the Adjusted R²
print("Adjusted R-squared (Adjusted R²):", adjusted_r2)

# Step 13: Prediction for new data (example: predicting height for a weight of 72 kg)
new_prediction = regression.predict(scaler.transform([[47]]))
print("Predicted Height for a weight of 72kg:", new_prediction)

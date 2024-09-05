import numpy as np
import os
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Define the path to the uploaded ZIP file and the extraction directory
zip_file_path = 'C:/Users/HP/Downloads/t2/youtube_data.zip'
extraction_dir = 'youtube_data'

# Extract the ZIP file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

youtube_data_dir = os.path.join(extraction_dir, 'youtube_data/')

# Load the all csv files into a DataFrames
ca_videos_path = os.path.join(youtube_data_dir, 'CAvideos.csv')
fr_videos_path = os.path.join(youtube_data_dir, 'FRvideos.csv')
in_videos_path = os.path.join(youtube_data_dir, 'INvideos.csv')
us_videos_path = os.path.join(youtube_data_dir, 'USvideos.csv')


# Load the datasets
ca_df = pd.read_csv(ca_videos_path)  
fr_df = pd.read_csv(fr_videos_path) 
in_df = pd.read_csv(in_videos_path) 
us_df = pd.read_csv(us_videos_path)

df = pd.concat([ca_df, fr_df, in_df, us_df], ignore_index=True)


# Preprocess the data
df['views_log'] = np.log1p(df['views'])
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['publish_hour'] = df['publish_time'].dt.hour
df['publish_dayofweek'] = df['publish_time'].dt.dayofweek

# Select features and target variable
features = ['likes', 'dislikes', 'comment_count', 'publish_hour', 'publish_dayofweek']
X = df[features].fillna(0)
y = df['views_log']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# This will remove any rows where the target variable is NaN
X_test = X_test[X_test.notnull()]
X_train = X_train[X_train.notnull()]
y_test = y_test[y_test.notnull()]
y_train = y_train[y_train.notnull()]


# Initialize the models
ada_boost = AdaBoostRegressor(random_state=42)
gradient_boost = GradientBoostingRegressor(random_state=42)

# Train the models
ada_boost.fit(X_train, y_train)
gradient_boost.fit(X_train, y_train)

# Make predictions
y_pred_ada = ada_boost.predict(X_test)
y_pred_gradient = gradient_boost.predict(X_test)

# Evaluate the models

# Calculate MAE, MSE, and R² for AdaBoost
mae_ada = mean_absolute_error(y_test, y_pred_ada) 
mse_ada = mean_squared_error(y_test, y_pred_ada) 
r2_ada = r2_score(y_test, y_pred_ada)


# Calculate MAE, MSE, and R² for GradientBoost
mae_gradient = mean_absolute_error(y_test, y_pred_gradient)
mse_gradient = mean_squared_error(y_test, y_pred_gradient)
r2_gradient = r2_score(y_test, y_pred_gradient)


# Print the metrics for comparison
print(f"Mean Absolute Error | for Gradient Boosting : {mae_gradient:.4f} & AdaBoost : {mae_ada:.4f}")
print(f"Mean Squared Error  | for Gradient Boosting : {mse_gradient:.4f} & AdaBoost : {mse_ada:.4f}")
print(f"R²                  | for Gradient Boosting : {r2_gradient:.4f} & AdaBoost : {r2_ada:.4f}")


# Plotting Bar Graph Comparison
plt.figure(figsize=(14, 6))

# MAE comparison
plt.subplot(1, 3, 1)  
bar_width = 0.35
index = np.arange(2)
bar1 = plt.bar(index, [mae_ada, mae_gradient], bar_width, label='MAE', color='b')

plt.xlabel('Model')
plt.ylabel('Error')
plt.title('Mean Absolute Error Comparison')
plt.xticks(index, ['AdaBoost', 'GradientBoost'])
plt.legend()


# MSE comparison
plt.subplot(1, 3, 2)  
bar2 = plt.bar(index, [mse_ada, mse_gradient], bar_width, label='MSE', color='r')

plt.xlabel('Model')
plt.ylabel('Error')
plt.title('Mean Squared Error Comparison')
plt.xticks(index, ['AdaBoost', 'GradientBoost'])
plt.legend()


# R2 comparison
plt.subplot(1, 3, 3)  
bar3 = plt.bar(index, [r2_ada, r2_gradient], bar_width, label='R²', color='g')

plt.xlabel('Model')
plt.ylabel('Error')
plt.title('R² Comparison')
plt.xticks(index, ['AdaBoost', 'GradientBoost'])
plt.legend()

plt.show()

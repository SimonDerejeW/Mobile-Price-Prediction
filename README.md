# 📱 Mobile Price Prediction

Predicting Mobile Phone Prices using Linear Regression. This project includes data preprocessing, model training, evaluation, and visualization to assess model performance and improve prediction accuracy.

## 📝 Project Overview

This project involves implementing a linear regression model trained on a dataset containing mobile phone hardware specifications and their prices. The trained model is used to predict the prices of mobile phones based on their features. Additionally, a simple Flask web application is provided to allow users to input phone specifications and get price predictions.

## ✨ Features

- **Data Preprocessing**: Cleaning and preparing the dataset for model training.
- **Model Training**: Training a linear regression model using scikit-learn.
- **Model Evaluation**: Evaluating the performance of the model using various metrics.
- **Visualization**: Visualizing the data and model performance.
- **Web Application**: A Flask web app for user interaction and price prediction.

## 🛠️ Setup and Installation

1. **Clone the repository**:
    ```
    git clone https://github.com/SimonDerejeW/Mobile-Price-Prediction.git
    cd Mobile-Price-Prediction
    ```

2. **Create and activate a virtual environment**:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```
    pip install -r requirements.txt
    ```

4. **Run the web application**:
    ```
    python app.py
    ```

## 🚀 Usage

1. **Data Preparation**:
    - Load the dataset and preprocess it.
    - Split the data into training and testing sets.

2. **Model Training and Evaluation**:
    - Train the linear regression model.
    - Evaluate the model's performance on the test set.

3. **Web Application**:
    - Access the web app at `http://localhost:5000` or if you just wanna take a look at the web app check it out [here](https://mobile-price-prediction-78by.onrender.com).
    - Enter the mobile phone specifications in the form.
    - Get the predicted price for the mobile phone.

## 📂 File Structure

- `app.py`: The main Flask web app script.
- `model.pkl`: The trained linear regression model.
- `scaler.pkl`: The scaler used for standardizing input features.
- `requirements.txt`: List of required Python packages.

## 📋 Requirements

- joblib
- numpy
- flask
- scikit-learn
- seaborn
- pandas

## 🤝 Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss any changes.

## 📜 License

This project is licensed under the MIT License.


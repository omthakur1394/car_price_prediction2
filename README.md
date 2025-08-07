# 🚗 Car Price Prediction

This project predicts the selling price of used cars using machine learning. It uses an XGBoost model trained on features like brand, mileage, engine, fuel type, and more. The model is deployed through a Streamlit web app.

---

## 🌐 Live Demo

🔗 **Try the app here**: [Car Price Predictor Streamlit App](https://carpriceprediction2-amvy7tfthcf68wzwmck5bk.streamlit.app/)

---

## 📁 Project Structure

| File         | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `main.py`    | Trains the model using XGBoost with log-transformed target                 |
| `main2.py`   | Loads saved model and pipeline, predicts from `input.csv`, saves `Output.csv` |
| `app.py`     | Streamlit app for live predictions                                         |
| `data.csv`   | Dataset used for model training                                            |
| `input.csv`  | Sample input data to be predicted in bulk                                  |

> ⚠️ Note: `model.pkl`, `pipeline.pkl`, and `Output.csv` are generated after running the scripts and are not included in the GitHub repo by default.

---

## 🧠 Features Used for Prediction

- Brand  
- Vehicle age  
- Kilometers driven  
- Seller type  
- Fuel type  
- Transmission type  
- Mileage  
- Engine capacity  
- Max power  
- Number of seats  

---

## 💡 How the Project Works

### 🔹 `main.py`
- Loads dataset (`data.csv`)
- Applies feature engineering and transformation (`np.log1p`)
- Trains XGBoost Regressor
- Saves model and preprocessing pipeline (`model.pkl`, `pipeline.pkl`)

### 🔹 `main2.py`
- Loads trained pipeline and model
- Reads input data from `input.csv`
- Makes predictions and writes results to `Output.csv`

### 🔹 `app.py`
- Uses Streamlit to create a user interface
- Takes user inputs via form
- Makes live predictions using the trained model
- Displays predicted price on the screen

---

## 🚀 How to Run the Project Locally

### ✅ Step 1: Clone the repository

```bash
git clone https://github.com/omthakur1394/car_price_prediction2.git
cd car_price_prediction2

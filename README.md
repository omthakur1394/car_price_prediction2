# 🚗 Car Price Prediction – XGBoost · Streamlit · FastAPI

This project predicts the **selling price of used cars** using an **XGBoost Regressor** trained on features like brand, mileage, engine, fuel type, and more.  
It includes:
- A **Streamlit web app** for live predictions
- A **FastAPI endpoint** for programmatic access
- Bulk prediction capability from CSV files

---

## 🌐 Live Demo
- **Streamlit App:** [Car Price Predictor](https://carpriceprediction2-amvy7tfthcf68wzwmck5bk.streamlit.app/)  
- **FastAPI Endpoint:** [API on Render](https://test4-afer.onrender.com/)

---

## 📂 Project Structure
| File | Description |
|------|-------------|
| `main.py` | Trains the model using XGBoost with log-transformed target |
| `main2.py` | Loads saved model and pipeline, predicts from `input.csv`, saves `Output.csv` |
| `app.py` | Streamlit app for live predictions |
| `data.csv` | Dataset used for model training |
| `input.csv` | Sample input data for bulk predictions |

⚠️ **Note:** `model.pkl`, `pipeline.pkl`, and `Output.csv` are generated after running scripts and are not included in the repo by default.

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

### Training & Prediction
- **`main.py`** → Loads dataset, applies preprocessing & feature engineering, trains **XGBoost** model, saves `model.pkl` and `pipeline.pkl`.  
- **`main2.py`** → Loads saved model, makes bulk predictions from `input.csv`, saves results to `Output.csv`.  
- **`app.py`** → Streamlit UI for live predictions.  

---

## 🚀 How to Run Locally
```bash
# 1️⃣ Clone the repository
git clone https://github.com/omthakur1394/Car-Price-Prediction-with-Streamlit-and-FastAPI.git
cd Car-Price-Prediction-with-Streamlit-and-FastAPI

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Train the model and save pipeline
python main.py

# 4️⃣ (Optional) Run bulk prediction from CSV
python main2.py

# 5️⃣ Start the Streamlit web app
streamlit run app.py




import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scalers
model = tf.keras.models.load_model("trained_plant_disease_model.keras")
with open('model.pkl', 'rb') as f:
    crop_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Function to predict crop disease based on image
def disease_prediction(image):
    img = Image.open(BytesIO(image))
    img = img.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Function to recommend crop based on input data
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    transformed_features = scaler.transform(features)
    prediction = crop_model.predict(transformed_features)

    crop_dict = {
        0: 'rice', 1: 'maize', 2: 'jute', 3: 'cotton', 4: 'coconut', 5: 'papaya', 6: 'orange',
        7: 'apple', 8: 'muskmelon', 9: 'watermelon', 10: 'grapes', 11: 'mango', 12: 'banana',
        13: 'pomegranate', 14: 'lentil', 15: 'blackgram', 16: 'mungbean', 17: 'mothbeans',
        18: 'pigeonpeas', 19: 'kidneybeans', 20: 'chickpea', 21: 'coffee'
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return result


# API endpoint for disease recognition
@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = disease_prediction(contents)

    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                   'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                   'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                   'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                   'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                   'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

    return {"prediction": class_names[prediction]}


# Pydantic model for crop recommendation input
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# API endpoint for crop recommendation
@app.post("/recommend_crop")
async def recommend_crop(crop_input: CropInput):
    result = recommendation(
        crop_input.N, crop_input.P, crop_input.K,
        crop_input.temperature, crop_input.humidity,
        crop_input.ph, crop_input.rainfall
    )
    return {"recommendation": result}


# Streamlit UI
def main():
    st.sidebar.title("BAZAFARM PREDICTIONS")
    st.sidebar.subheader("Predictions")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "Crop Recommendation", "About"])

    if app_mode == "Home":
        st.header("BAZAFARM PREDICTION SYSTEM")

        image_path = "home_page.jpeg"
        st.image(image_path, use_column_width=True)
        st.markdown("""
        Welcome to the BAZAFARM Plant Disease Recognition and Crop Recommendation System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Crop Recommendation
        - **Precision Agriculture:** Utilizing AI and data analytics to recommend the best crops based on soil and climate conditions.
        - **Tailored Solutions:** Customized crop suggestions to optimize yield and sustainability.
        - **Expert Advice:** Access insights from agricultural specialists for informed decision-making.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        """)

    elif app_mode == "Disease Recognition":
        st.header("Disease Recognition")
        test_image = st.file_uploader("Choose an Image:")
        if st.button("Show Image"):
            st.image(test_image, width=4, use_column_width=True)
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            contents = test_image.read()
            prediction = disease_prediction(contents)
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                           'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                           'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                           'Corn_(maize)___healthy',
                           'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                           'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                           'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                           'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                           'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                           'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']
            st.success(f"Model is Predicting it's a {class_names[prediction]}")

    elif app_mode == "Crop Recommendation":
        st.header("Crop Recommendation")
        st.subheader("Enter the following details to get crop recommendation:")
        N = st.number_input("Nitrogen (N)", min_value=0)
        P = st.number_input("Phosphorus (P)", min_value=0)
        K = st.number_input("Potassium (K)", min_value=0)
        temperature = st.number_input("Temperature", min_value=0.0)
        humidity = st.number_input("Humidity", min_value=0.0)
        ph = st.number_input("pH", min_value=0.0)
        rainfall = st.number_input("Rainfall", min_value=0.0)

        if st.button("Recommend Crop"):
            recommended_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)
            st.success(f"Recommended crop: {recommended_crop}")

    elif app_mode == "About":
        st.header("About BAZAFARM Technology")
        st.markdown("""
        **BAZAFARM TECHNOLOGY**

        **WHAT IS BAZAFARM?**
        BAZAFARM is a solar-powered IoT device designed to measure water level, soil temperature, and soil fertility in real-time on farms. Farmers can access this data via their mobile phones, tablets, or PCs over the Internet and make informed decisions to optimize crop yields. The device helps farmers achieve high crop yields by providing actionable insights into soil conditions.

        **HOW DOES BAZAFARM WORK?**
        BAZAFARM devices utilize IoT technology and operate within a network of sensor nodes that transmit data to a central hub, which then sends it to the Internet. For farms smaller than 5 hectares, a standalone device capable of measuring and transmitting data to the Internet is sufficient.

        **BENEFITS OF BAZAFARM**
        - Optimizes water and fertilizer usage.
        - Improves crop production quality.
        - Enables collection and storage of big data for informed decision-making.
        - Provides access to weather forecasts for farmers.
        """)


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="0.0.0.0", port=8000)
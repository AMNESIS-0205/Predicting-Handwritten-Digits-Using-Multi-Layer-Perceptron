import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2

# --- Page Configuration ---
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

# --- Load Model ---
# Using the full path ensures the cloud server finds the 'brain' we trained
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_my_model()

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🧠 Project Details")
st.sidebar.info(
    "This Multi-Layer Perceptron (MLP) was built to recognize handwritten digits (0-9). "
    "It demonstrates high accuracy image classification without using CNNs."
)
st.sidebar.markdown("### 📊 Metrics")
st.sidebar.write("- **Model:** MLP (Dense Layers)")
st.sidebar.write("- **Dataset:** MNIST")
st.sidebar.write("- **Optimizer:** Adam")
st.sidebar.divider()
st.sidebar.markdown("Created by: **Arpit Malviya**")

# --- Main UI ---
st.title("🔢 Handwritten Digit Recognizer")
st.write("Draw a digit in the box below and click **Predict** to see the model's thought process.")

# Create two columns for a dashboard feel
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖋️ Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White ink
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300, 
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button('Clear Canvas'):
        st.rerun()

with col2:
    st.subheader("🔍 Prediction Results")
    
    if canvas_result.image_data is not None:
        # Preprocessing: Resize to 28x28 and grayscale to match training data
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Reshape for MLP (Flatten to 784) and Normalize
        img_input = img.reshape(1, 784) / 255.0
        
        if st.button('Predict Digit'):
            # Get probabilities for all 10 digits
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Display Big Result
            st.metric(label="Predicted Digit", value=predicted_digit)
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Visualizing the "Thinking Process"
            st.write("### 📈 Confidence Levels (0-9)")
            # Create a bar chart showing probabilities for each class
            chart_data = prediction[0]
            st.bar_chart(chart_data)

# --- Footer ---
st.divider()
st.caption("Developed as part of the Data Science Portfolio Project.")

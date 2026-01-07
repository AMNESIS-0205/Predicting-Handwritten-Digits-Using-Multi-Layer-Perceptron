import streamlit as st
import tensorflow as tf  # Use this for loading the model
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2

# 1. Load your trained "brain"
# Use the full path so the IDE and Cloud server can find it easily
model = tf.keras.models.load_model('mnist_model.h5')

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the box below!")

# 2. Create the Drawing Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # White ink
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 3. Process the drawing when the user is done
if canvas_result.image_data is not None:
    # Resize drawing to 28x28 pixels (matching MNIST)
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    # Convert to grayscale and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 784) / 255.0
    
    if st.button('Predict'):
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        st.header(f"Result: {digit}")
        st.bar_chart(prediction[0]) # Show probabilities for all 10 digits
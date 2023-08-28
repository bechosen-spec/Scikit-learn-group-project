import streamlit as st
import joblib

# Load the saved model
loaded_model = joblib.load("my_model.joblib")

# Streamlit UI components
st.set_page_config(page_title="Model Prediction App", page_icon="🔮")

# CSS styles
st.markdown(
    """
    <style>
    .stButton {
        background-color: #007BFF;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main content
st.title("Conflict Prediction App")
st.write("Enter the values for start_prec and start_prec2 to predict target class.")

# Input fields for features
start_prec = st.number_input("start_prec", value=0.0)
start_prec2 = st.number_input("start_prec2", value=0.0)

# Predict button
if st.button("Predict", key="predict_button", help="Click to make a prediction"):
    # Create a feature vector from user inputs
    feature_vector = [[start_prec, start_prec2]]
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(feature_vector)
    
    # Display the prediction result
    st.markdown(f"**Predicted class:** {prediction[0]}")

# Footer
st.markdown("---")
st.write("Created with ❤️ by Dipeolu")

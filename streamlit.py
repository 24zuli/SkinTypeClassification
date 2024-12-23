import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import tempfile
import os
model = load_model('skin_type_model.h5')

# Example class_indices dictionary
class_indices = {
    0: 'normal',
    1: 'oily',
    2: 'dry',
    3: 'oily/acne',
    4: 'dry/acne',
    # Add other classes as necessary
}

# Function to recommend skincare ingredients based on predictions
def recommend_products(image_path, model, class_indices):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict skin type and concerns
    prediction = model.predict(img_array)
    predicted_class = class_indices[np.argmax(prediction)]

    # Log the predicted class
    st.write(f"Predicted class: {predicted_class}")

    # Initialize skin_type and skin_concern
    skin_type = predicted_class
    skin_concern = None

    # Parse the predicted class to get skin type and concern if applicable
    if '/' in predicted_class:
        skin_type, skin_concern = predicted_class.split('/')

    # Recommendations based on predictions
    recommendations = {
        'skin_type': skin_type,
        'skin_concern': skin_concern,
        'recommended_ingredients': []
    }

    # Example recommendation logic
    if skin_type == 'oily':
        recommendations['recommended_ingredients'].extend(['Salicylic Acid', 'Niacinamide'])
    if skin_concern == 'acne':
        recommendations['recommended_ingredients'].append('Benzoyl Peroxide')
    if skin_type == 'normal':
        recommendations['recommended_ingredients'].extend(['Hyaluronic Acid', 'Glycerin'])

    return recommendations

# Streamlit app
def main():
    st.title("Skincare Ingredients Recommendation System")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Get recommendations
        recommendations = recommend_products(temp_file_path, model, class_indices)

        # Display recommendations
        st.write("### Recommendations:")
        st.write(f"**Skin Type:** {recommendations['skin_type']}")
        if recommendations['skin_concern']:
            st.write(f"**Skin Concern:** {recommendations['skin_concern']}")
        st.write(f"**Recommended Ingredients:** {', '.join(recommendations['recommended_ingredients'])}")

if __name__ == '__main__':
    main()

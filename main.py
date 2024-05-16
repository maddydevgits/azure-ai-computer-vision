import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io

# Add your Computer Vision subscription key and endpoint
subscription_key = ""
endpoint = ""

# Authenticate the client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

def object_recognition(image):
    # Analyze the image for objects
    objects = []
    results = computervision_client.detect_objects_in_stream(image)
    for obj in results.objects:
        objects.append(obj.object_property)
    return objects

def ocr_extraction(image):
    # Extract text using OCR
    ocr_results = computervision_client.recognize_printed_text_in_stream(image, language="en")
    extracted_text = ""
    for region in ocr_results.regions:
        for line in region.lines:
            extracted_text += " ".join([word.text for word in line.words]) + "\n"
    return extracted_text

def generate_caption(image):
    # Analyze the image for visual features
    features = [
        VisualFeatureTypes.description,
        VisualFeatureTypes.tags,
        VisualFeatureTypes.objects,
        VisualFeatureTypes.categories
    ]
    analysis_results = computervision_client.analyze_image_in_stream(image, visual_features=features)

    # Generate prompt based on analysis
    prompt = ""
    if analysis_results.description and analysis_results.description.captions:
        prompt += "Description: " + analysis_results.description.captions[0].text + "\n"
    # if analysis_results.tags:
    #     prompt += "Tags: " + ", ".join([tag.name for tag in analysis_results.tags]) + "\n"
    # if analysis_results.objects:
    #     prompt += "Objects: " + ", ".join([obj.object_property for obj in analysis_results.objects]) + "\n"
    # if analysis_results.categories:
    #     prompt += "Categories: " + ", ".join([category.name for category in analysis_results.categories]) + "\n"
    
    return prompt

def main():
    st.title("Image Analysis App")

    # Sidebar navigation
    page_options = ["Object Recognition", "OCR Extraction", "Generate Caption"]
    page_selection = st.sidebar.selectbox("Select Option", page_options)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analyze the image based on user selection
        if page_selection == "Object Recognition":
            if st.button("Recognize Objects"):
                with st.spinner('Recognizing Objects...'):
                    # Convert image to stream
                    image_stream = io.BytesIO()
                    # Convert image to RGB mode
                    image = image.convert("RGB")
                    image.save(image_stream, format='JPEG')
                    image_stream.seek(0)
                    # Recognize objects in the image
                    objects = object_recognition(image_stream)
                # Display recognized objects
                st.write("Recognized Objects:", ", ".join(objects))

        elif page_selection == "OCR Extraction":
            if st.button("Extract Text"):
                with st.spinner('Extracting Text...'):
                    # Convert image to stream
                    image_stream = io.BytesIO()
                    # Convert image to RGB mode
                    image = image.convert("RGB")
                    image.save(image_stream, format='JPEG')
                    image_stream.seek(0)
                    # Extract text using OCR
                    extracted_text = ocr_extraction(image_stream)
                # Display extracted text
                st.write("Extracted Text:\n", extracted_text)

        elif page_selection == "Generate Caption":
            if st.button("Generate Caption"):
                with st.spinner('Generating Caption...'):
                    # Convert image to stream
                    image_stream = io.BytesIO()
                    # Convert image to RGB mode
                    image = image.convert("RGB")
                    image.save(image_stream, format='JPEG')
                    image_stream.seek(0)
                    # Generate caption based on image analysis
                    prompt = generate_caption(image_stream)
                # Display the generated prompt
                st.success(prompt)

if __name__ == "__main__":
    main()

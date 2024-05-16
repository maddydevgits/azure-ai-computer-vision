from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os

# Add your Computer Vision subscription key and endpoint to your environment variables
subscription_key = ""
endpoint = ""

# Authenticate the client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Path to the local image file
image_path = "sai.jpeg"

with open(image_path, "rb") as image_stream:
    # Analyze the image
    print("Analyzing local image...")
    features = [
        VisualFeatureTypes.description,
        VisualFeatureTypes.tags,
        VisualFeatureTypes.objects,
        VisualFeatureTypes.categories
    ]
    results = computervision_client.analyze_image_in_stream(image_stream, visual_features=features)
    # print(results)

    # Print results
    if results.description and results.description.captions:
        print("Description: ", results.description.captions[0].text)
    else:
        print("No description available.")

    print("Tags: ", [tag.name for tag in results.tags])
    print("Objects: ", [obj.object_property for obj in results.objects])
    print("Categories: ", [category.name for category in results.categories])

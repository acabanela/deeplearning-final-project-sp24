import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2

# Load the PyTorch model
model_path = './vit_v1_model.pth'

# Define the model architecture
model = models.vit_b_16(pretrained=False)
num_classes = 4  # Number of classes
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

# Load the PyTorch model and its state dictionary
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.43613595, 0.4974372, 0.3781651],
                         std=[0.21189487, 0.22010513, 0.21154968]),  # Normalize using mean and std
])

class_info = {
    0: {
        'Name': 'Cercospora leaf spot (Gray leaf spot)',
        'Scientific Name': 'Cercospora zeae-maydis',
        'Symptoms': [
            'Small necrotic spots with chlorotic halos on leaves which expand to rectangular lesions 1-6 cm in length and 2-4 mm wide; as the lesions mature they turn tan in color and finally gray; lesions have sharp, parallel edges and are opaque; disease can develop quickly causing complete blighting of leaves and plant death.',
            'Brown Spots with yellow rings throughout the leaf during the growing period of the Cassava',
            'Lesions that are 0.15-0.2 cm in diameter',
            'Serious cases can lead to holes throughout the lesions on the leaf'
        ],
        'Cause': 'Fungus',
        'Comments': 'Disease emergence is favored in areas where a corn crop is followed by more corn with no rotation; severity and incidence of disease is likely due to continuous corn culture with minimum tillage and the use of susceptible hybrids in in the midwestern corn belt of the USA; prolonged periods of foggy or cloudy weather can cause severe Cercopora epidemics.',
        'Management': 'Plant corn hybrids with resistance to the disease; crop rotation and plowing debris into soil may reduce levels of inoculum in the soil but may not provide control in areas where the disease is prevalent; foliar fungicides may be economically viable for some high yielding susceptible hybrids.',
        'Source': 'This information has been collected from the Corn-Maize webpage of the PlantVillage platform. Visit this page for more information: [PlantVillage Maize (Corn) Page](https://plantvillage.psu.edu/topics/corn-maize/infos)'
    },
    1: {
        'Name': 'Common rust',
        'Scientific Name': 'Puccinia sorghi',
        'Symptoms': [
            'Oval or elongated cinnamon brown pustules on upper and lower surfaces of leaves; pustules rupture and release powdery red spores; pustules turn dark brown-black as they mature and release dark brown powdery spores; if infection is severe, pustules may appear on tassels and ears and leaves may begin to yellow; in partially resistant corn hybrids, symptoms appear as chlorotic or necrotic flecks on the leaves which release little or no spore.'
        ],
        'Cause': 'Fungus',
        'Comments': 'Disease is spread by wind-borne spores; some of the most popularly grown sweet corn varieties have little or no resistance to the disease.',
        'Management': 'The most effective method of controlling the disease is to plant resistant hybrids; application of appropriate fungicides may provide some degree on control and reduce disease severity; fungicides are most effective when the amount of secondary inoculum is still low, generally when plants only have a few rust pustules per leaf.',
        'Source': 'This information has been collected from the Corn-Maize webpage of the PlantVillage platform. Visit this page for more information: [PlantVillage Maize (Corn) Page](https://plantvillage.psu.edu/topics/corn-maize/infos)'
    },
    2: {
        'Name': 'Northern Leaf Blight',
        'Scientific Name': 'Exserohilum turcicum',
        'Symptoms': [
            'In the beginning we will notice elliptical gray-green lesions on leaves. As the disease process this lesions become pale gray to tan color. Later stage the lesions looks dirty due to dark gray spores particularly under lower leaf surface. The disease can be easily identified in the field due to its long, narrow lesions which are unrestricted by veins.'
        ],
        'Cause': 'Fungus',
        'Comments': 'The disease mainly spread through rain splash and wind.',
        'Management': 'Follow proper tillage to reduce fungus inoculum from crop debris. Follow crop rotation with non host crop. Grow available resistant varieties. In severe case of disease incidence apply suitable fungicide.',
        'Source': 'This information has been collected from the Corn-Maize webpage of the PlantVillage platform. Visit this page for more information: [PlantVillage Maize (Corn) Page](https://plantvillage.psu.edu/topics/corn-maize/infos)'
    },
    3: {
        'Name': 'Healthy',
        'Symptoms': [ 'No visible symptoms of disease are observed on the crop.' ],
        'Cause': 'N/A',
        'Comments': 'The crop is free from any diseases or abnormalities.',
        'Management': 'Continue with regular crop maintenance practices to ensure crop health and productivity.',
        'Source': 'This information has been collected from the Corn-Maize webpage of the PlantVillage platform. Visit this page for more information: [PlantVillage Maize (Corn) Page](https://plantvillage.psu.edu/topics/corn-maize/infos)'
    }
}

# Streamlit interface
st.title('Corn Crop Disease Diagnosis App')
uploaded_file = st.file_uploader("Upload a leaf image from a corn crop below:", type=['png', 'jpg', 'jpeg', 'JPG'])

# Sidebar
with st.sidebar:
    st.subheader("About the Uploaded Leaf")
    if uploaded_file is None:
        st.write("No information to show yet.\n\nUpload an image of a corn crop leaf on the right to get started! 👉 ")
    else:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        image = image.resize((224, 224), Image.BILINEAR)  # Resize using linear interpolation
        image = transforms.ToTensor()(image)

        # Run model inference
        with torch.no_grad():
            prediction = model(image.unsqueeze(0))
            _, predicted_class = torch.max(prediction, 1)

        # Map predicted class index to label
        class_names = ['Gray Leaf Spot',
                       'Common Rust',
                       'Northern Leaf Blight',
                       'Healthy']
        result = class_names[predicted_class.item()]

        # Display predicted class result
        if result == 'Healthy':
            st.success(f'Your leaf is {result}!')
        else:
            st.warning(f'Your leaf has this disease: {result}')

        # Display class information
        class_info_key = class_names.index(result)
        class_details = class_info.get(class_info_key, {})
        for key, value in class_details.items():
            st.subheader(key)
            if isinstance(value, list):
                for item in value:
                    st.write(item)
            else:
                st.write(value)

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Crop Image', width=300)

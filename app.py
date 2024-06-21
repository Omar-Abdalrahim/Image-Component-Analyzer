import streamlit as st
from PIL import Image, ImageDraw, ImageFont  # for uploading the image and show the lables on the image
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="Image Component Analyzer",
    page_icon="🖼️",
    layout="centered",
)

confidence_threshold=0.5

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    return model

model = load_model()

def preprocess_image(image):
    # Convert the image to RGB format
    image = image.convert("RGB")
    return image

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def analyze_image(image, model):
    image = image.convert("RGB")     # Convert image to RGB format
        
    image_np = np.array(image)       # Convert image to numpy array
    
    result = model.predict(image)[0] # Perform object detection
    
    # Convert image back to PIL Image
    image = Image.fromarray(image_np)
    
    # Draw on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    detected_labels = set()

    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        pos = box.xyxy[0].tolist()
        coords  = [round(x) for x in pos]
        Confedence = round(box.conf[0].item(), 2)
        
        if Confedence >= confidence_threshold:
            detected_labels.add((class_id, Confedence))
            
            # Draw rectangle and add label
            draw.rectangle(coords , outline="red", width=3)
            label = f"{class_id}: {Confedence}"
            
            Size = textsize(label, font=font)
            draw.rectangle([coords [0], coords [1] - Size[1], coords [0] + Size[0], coords [1]], fill="red")
            draw.text((coords [0], coords [1] - Size[1]), label, fill="white", font=font)
            
    return list(detected_labels), image



st.write('Upload an image and click "Analyze Image" to see the components detected.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image'):
        st.write('Analyzing...')
        preprocessed_image = preprocess_image(image)
        
        detected_labels,Processed = analyze_image(preprocessed_image, model)
        detected_labels.sort()
        
        st.image(Processed, caption='Processed Image.', use_column_width=True)
        st.write("Detected components:")
        for label in detected_labels:
            st.write(label)
            

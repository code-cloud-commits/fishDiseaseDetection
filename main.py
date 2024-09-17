import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO

# Load the YOLOv9 model
model_path = 'best.pt'  # Update this with the path to your YOLOv9 model
model = YOLO(model_path)





st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


#Main Page
if(app_mode=="Home"):
    video_url = 'https://media.istockphoto.com/id/1158105530/video/colorful-siamese-halfmoon-fighting-betta-fish-white-red-color-black-background.mp4?s=mp4-640x640-is&k=20&c=aUD_jfQqp9blHfSjuzB50jyICZAXHR3hSMfh5KtDRtM='

    st.markdown(f"""
    <style>
    .video-background {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -2;
    }}
    .image-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
    }}
    .stApp {{
        position: relative;
        z-index: 1;
    }}
    </style>
    <video autoplay muted loop class="video-background">
        <source src="{video_url}" type="video/mp4">
    </video>
    """, unsafe_allow_html=True)

    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "pexels-photo-8691140.jpg"
    st.image(image_path,use_column_width=True)
    
    st.markdown("""
    Welcome to the Fish Disease Detection System! 🐟🔍

    Our mission is to efficiently identify diseases in fish through advanced image analysis. Upload an image of a fish, and our system will analyze it to detect any signs of diseases. Let’s ensure healthier aquatic life together!

    ###How It Works

    1. **Upload Image:** Navigate to the Disease Detection page and upload an image of the fish showing potential signs of disease.
    2. **Analysis:** Our system uses cutting-edge YOLOv9-based machine learning models to process the image and detect possible diseases.
    3. **Results:** Receive detailed results and recommendations for further action.

    ###Why Choose Us?

    - ***Accuracy:*** Our system leverages state-of-the-art YOLOv9 algorithms for precise disease detection.
    - ***User-Friendly:*** Enjoy a simple and intuitive interface for a smooth user experience.
    - ***Fast and Efficient:*** Get results in seconds for quick and informed decision-making.
    ###Get Started
    Click on the **Disease Detection** in the sidebar to upload an image and experience the power of our Fish Disease Detection System!

    ###About Us 
    Discover more about our project, our team, and our mission on the **About** page.
    """)
    
#About Project
elif(app_mode=="About"):
    st.header("About Us")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

    
elif(app_mode=="Disease Recognition"):
    class_names = {
        0: 'EUS',
        1: 'Eye Disease',
        2: 'Fin Lesions',
        3: 'Rotten Gills',
        # Add other classes as needed
        }
    def resize_image(image, target_size):
        """Resize the image to the target size with padding if needed."""
        width, height = image.size
        new_width, new_height = target_size
         # Resize and pad the image
        image = image.resize((new_width, new_height), Image.BICUBIC)
        return image
    def predict(image):
        # Convert image to tensor
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
        # Perform inference
        results = model(img_tensor)
        return results

    def process_predictions(results):
        # Assuming the results are in a list of predictions
        boxes = []
        scores = []
        classes = []
    
        for result in results:
            boxes.extend(result.boxes.xyxy.tolist())  # List of boxes in [x1, y1, x2, y2]
            scores.extend(result.boxes.conf.tolist())  # List of scores
            classes.extend(result.boxes.cls.tolist())  # List of class labels
    
        return boxes, scores, classes

    def draw_boxes(image, boxes, scores, classes):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            class_name = class_names.get(class_id, 'Unknown')  # Get the class name or 'Unknown'
            label = f'{class_name}: {scores[i]:.2f}'
            color = (255, 0, 0)  # Blue
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def main():
        st.title('Fish Disease Detection with YOLOv9')

        # Add a container to ensure that Streamlit content is positioned correctly
        with st.container():
         # Sidebar for file upload
            st.sidebar.header('Upload File')
            uploaded_file = st.sidebar.file_uploader("Choose an image or video file", type=['jpg', 'jpeg', 'png', 'mp4'])
        
            if uploaded_file:
                if uploaded_file.type in ['video/mp4']:
                    # Save the uploaded video to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_file_path = tmp_file.name
                
                    # Process video
                    cap = cv2.VideoCapture(temp_file_path)
                
                    st.sidebar.video(temp_file_path)
                
                    # Placeholder for video output
                    st.sidebar.header('Video Processing Results')
                    output_video_path = 'output_video.mp4'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
                
                    stframe = st.empty()
                
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    
                        # Convert frame to PIL Image and resize
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        pil_image = resize_image(pil_image, (640, 640))  # Resize to 640x640
                    
                        # Predict
                        results = predict(pil_image)
                        boxes, scores, classes = process_predictions(results)
                    
                        # Convert PIL Image back to OpenCV format
                        frame_resized = np.array(pil_image)
                        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                    
                        # Draw boxes on frame
                        frame_with_boxes = draw_boxes(frame_resized, boxes, scores, classes)
                        out.write(frame_with_boxes)
                    
                        # Display the frame
                        stframe.image(frame_with_boxes, channels='BGR', use_column_width=True)
                
                    cap.release()
                    out.release()
                    st.sidebar.success(f"Video processed and saved as {output_video_path}")
            
                else:
                    # Process image
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                
                    # Resize image
                    image = resize_image(image, (640, 640))  # Resize to 640x640
                
                    # Predict
                    results = predict(image)
                    boxes, scores, classes = process_predictions(results)
                
                    # Convert image to OpenCV format
                    image_cv = np.array(image)
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                
                    # Draw boxes on image
                    image_with_boxes = draw_boxes(image_cv, boxes, scores, classes)
                
                    # Display result
                    st.image(image_with_boxes, caption='Processed Image with Bounding Boxes', channels='BGR', use_column_width=True)
                
                    # Save processed image
                    output_image_path = 'output_image.jpg'
                    cv2.imwrite(output_image_path, image_with_boxes)
                    st.success(f"Image processed and saved as {output_image_path}")
                
                    # Show details
                    st.write("### Detection Details")
                    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                        class_name = class_names.get(int(cls), 'Unknown')  # Get the class name or 'Unknown'
                        st.write(f"**Class:** {class_name}")
                        st.write(f"**Confidence:** {score:.2f}")
                        st.write(f"**Bounding Box:** {box}")

    if __name__ == "__main__":
        main()


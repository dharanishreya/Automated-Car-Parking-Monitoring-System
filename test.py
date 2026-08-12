import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os

# Dimensions of the bounding boxes
width, height = 120, 50

# Load annotations from the pickle file
with open('carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

# Load the input video
video_path = 'carPark.mp4'  # Replace with your video file path
if not os.path.exists(video_path):
    raise FileNotFoundError(f"The video file '{video_path}' was not found. Please provide the correct video file.")

# Load the trained model (replace 'model_final.h5' with your actual model file)
model_path = 'model_final.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError("The model file 'model_final.h5' was not found. Please provide the trained model.")
model = load_model(model_path)

# Function to preprocess cropped images for the model
def preprocess_image(cropped_img):
    resized_img = cv2.resize(cropped_img, (48, 48))  # Resize to the model's input size (48x48)
    normalized_img = resized_img / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_img, axis=0)  # Add batch dimension

# Function to resize video frame while maintaining aspect ratio
def resize_frame_with_padding(frame, target_width=1280, target_height=720):
    h, w = frame.shape[:2]
    
    # Compute scale factor to resize the image
    scale = min(target_width / w, target_height / h)
    
    # Resize the frame while maintaining the aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # Calculate padding for width and height
    top_padding = (target_height - new_h) // 2
    bottom_padding = target_height - new_h - top_padding
    left_padding = (target_width - new_w) // 2
    right_padding = target_width - new_w - left_padding
    
    # Add padding to the resized frame
    padded_frame = cv2.copyMakeBorder(resized_frame, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return padded_frame

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_counter = 0  # Counter to track frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    
    # Skip every 20th frame (process only frames that are multiples of 20)
    if frame_counter % 20 != 0:
        continue  # Skip this frame and go to the next one

    # Initialize counts for each frame
    total_parked = 0  # Reset parked count for each frame
    total_not_parked = 0  # Reset not parked count for each frame

    # Resize the frame with padding to maintain the aspect ratio
    frame_resized = resize_frame_with_padding(frame)

    # Iterate over all bounding boxes and predict whether it contains a car
    for i, pos in enumerate(positionList):
        x, y = pos
        x2, y2 = x + width, y + height

        # Crop the region of interest (ROI) from the resized frame
        cropped_img = frame_resized[y:y2, x:x2]

        # Preprocess the cropped image for the model
        preprocessed_img = preprocess_image(cropped_img)

        # Predict using the trained model
        prediction = model.predict(preprocessed_img)
        is_car = prediction[0][0] > 0.5  # Assuming binary classification (car: 1, not car: 0)

        # Annotate the frame with the prediction
        color = (0, 255, 0) if is_car else (0, 0, 255)  # Green for car, Red for not car
        label = "Not Parked" if is_car else "Parked"

        # Update counts for parked and not parked for the current frame
        if label == "Not Parked":
            total_not_parked += 1
        else:
            total_parked += 1

        # Draw the bounding box
        cv2.rectangle(frame_resized, (x, y), (x2, y2), color, 2)

        # Place the label just below the bounding box
        cv2.putText(frame_resized, label, (x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resized frame with predictions
    cv2.putText(frame_resized, f'Parked: {total_parked} | Not Parked: {total_not_parked}', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Resized Video with Padding and Predictions", frame_resized)

    # Press 'q' to exit the video window
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

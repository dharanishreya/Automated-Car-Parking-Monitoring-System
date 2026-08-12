from flask import Flask, jsonify, render_template, Response
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Dimensions of the bounding boxes
width, height = 120, 50

# Dynamically set working directory (if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Load annotations from the pickle file
pickle_file_path = os.path.join(current_dir, 'carposition.pkl')
if not os.path.exists(pickle_file_path):
    raise FileNotFoundError(f"Pickle file '{pickle_file_path}' not found.")

with open(pickle_file_path, 'rb') as f:
    positionList = pickle.load(f)

# Load the trained model
model_path = os.path.join(current_dir, 'model_final.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' was not found. Please provide the trained model.")

model = load_model(model_path)

# Function to preprocess cropped images for the model
def preprocess_image(cropped_img):
    resized_img = cv2.resize(cropped_img, (48, 48))  # Resize to the model's input size (48x48)
    normalized_img = resized_img / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_img, axis=0)  # Add batch dimension

# Function to resize video frame while maintaining aspect ratio
def resize_frame_with_padding(frame, target_width=1280, target_height=720):
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))

    top_padding = (target_height - new_h) // 2
    bottom_padding = target_height - new_h - top_padding
    left_padding = (target_width - new_w) // 2
    right_padding = target_width - new_w - left_padding

    padded_frame = cv2.copyMakeBorder(
        resized_frame, 
        top_padding, bottom_padding, left_padding, right_padding, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return padded_frame

# Function to check parking space and annotate the frame
def check_parking_space(frame):
    total_parked = 0
    total_not_parked = 0

    # Resize the frame with padding
    frame_resized = resize_frame_with_padding(frame)

    # Iterate over all bounding boxes and predict whether it contains a car
    for pos in positionList:
        x, y = pos
        x2, y2 = x + width, y + height

        # Crop the region of interest (ROI)
        cropped_img = frame_resized[y:y2, x:x2]

        # Preprocess the cropped image for the model
        preprocessed_img = preprocess_image(cropped_img)

        # Predict using the trained model
        prediction = model.predict(preprocessed_img)
        is_car = prediction[0][0] > 0.5  # Assuming binary classification (car: 1, not car: 0)

        # Annotate the frame with the prediction
        label = "Not Parked" if is_car else "Parked"

        # Update counts for parked and not parked for the current frame
        if label == "Not Parked":
            total_not_parked += 1
        else:
            total_parked += 1

        # Draw the bounding box and label on the frame
        color = (0, 255, 0) if is_car else (0, 0, 255)
        cv2.rectangle(frame_resized, (x, y), (x2, y2), color, 2)
        cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame_resized, total_parked, total_not_parked

# Route to display the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Function to generate frames for video streaming
def generate_frames():
    video_path = os.path.join(current_dir, 'carPark.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    cap = cv2.VideoCapture(video_path)  # Path to the video file
    frame_counter = 0  # Initialize frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Process only every 20th frame
        if frame_counter % 20 == 0:
            # Check parking space and get annotated frame
            frame_annotated, total_parked, total_not_parked = check_parking_space(frame)

            ret, buffer = cv2.imencode('.jpg', frame_annotated)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get parking space status (free vs occupied)
@app.route('/space_count')
def space_count():
    def generate():
        video_path = os.path.join(current_dir, 'carPark.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")

        cap = cv2.VideoCapture(video_path)
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_counter += 1

                # Process only every 20th frame
                if frame_counter % 20 == 0:
                    _, total_parked, total_not_parked = check_parking_space(frame)
                    
                    # Yield updated values as server-sent events (SSE)
                    yield f"data: {{\"free\": {total_not_parked}, \"occupied\": {total_parked}}}\n\n"
            else:
                break
        
        cap.release()

    # Return a streaming response to the client
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import numpy as np

# Initialize the face detection model (MTCNN) and recognition model (FaceNet)
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()


# model.prepare(ctx_id=-1) face_recg2\face_recog_deepNN\dnn_face_recog\videos\harry_potter_premier.mp4
# face_recg2\face_recog_deepNN\dnn_face_recog\videos\received_1179123699699768.mp4 face_recg2\face_recog_deepNN\dnn_face_recog\videos\face-demographics-walking.mp4
video_path = 'videos/received_1179123699699768.mp4'  # Replace with your video file pathface_recg2\face_recog_deepNN\dnn_face_recog\videos\classroom.mp4
# video_capture = cv2.VideoCapture(0)
rtsp_url="rtsp://admin:Sscl1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# video_capture = cv2.VideoCapture(video_path)
video_capture = cv2.VideoCapture(rtsp_url)



# Optional: Add known embeddings for comparison (replace these with actual embeddings)
known_face_embeddings = {
    "Person 1": np.random.rand(512),  # Random example embeddings
    "Person 2": np.random.rand(512)
}

frame_skip = 20  # Process every 2nd frame
frame_count = 0


# Main loop for capturing video
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    # Convert the image to RGB (OpenCV loads images in BGR format)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    boxes, probs = mtcnn.detect(img_rgb)
    # print(boxes)
    print(probs)

    if boxes is not None:
        for i, box in enumerate(boxes):
            # Draw bounding box
            cv2.rectangle(frame, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          (0, 255, 0), 2)
            
            # Extract the face from the frame and pass it to FaceNet
            face = mtcnn(img_rgb)[i]
            if face is not None:
                # Get the face embedding from FaceNet
                embedding = model(face.unsqueeze(0))
                print(len(embedding[0]))
                # Compare the detected face embedding to known face embeddings
                # min_distance = float('inf')
                # recognized_name = "Unknown"
                # for name, known_embedding in known_face_embeddings.items():
                #     distance = cosine(embedding.detach().numpy(), known_embedding)
                #     if distance < 0.6:  # You can adjust this threshold
                #         recognized_name = name
                #         break
                
                # Display the name of the recognized person on the video frame
                cv2.putText(frame, "Person", 
                            (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            (0, 255, 0), 2)
    
    # Display the video frame with bounding boxes and recognized names
    cv2.imshow('Face Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()

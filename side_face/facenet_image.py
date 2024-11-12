import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
import numpy as np

# Initialize MTCNN (face detection model) and InceptionResnetV1 (FaceNet recognition model)
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load an image (side view or frontal)
img1 = cv2.imread('images/rdj_front.jpg')
img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

img2 = cv2.imread('images/rdj_front.jpg')
img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# Detect faces
boxes, probs = mtcnn.detect(img_rgb1)
print(probs)
box, prob = mtcnn.detect(img_rgb2)
print(prob)

# Show the detected faces
# plt.figure(figsize=(10, 10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()

embedding1=[]

if boxes is not None:
    for i in range(boxes.shape[0]):
        cv2.rectangle(img_rgb1, 
                      (int(boxes[i][0]), int(boxes[i][1])), 
                      (int(boxes[i][2]), int(boxes[i][3])), 
                      (0, 255, 0), 2)  # Draw bounding box around detected face
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb1)
    plt.axis('off')
    plt.show()
    
    # Extract face embeddings
    faces = mtcnn(img_rgb1)  # Detect faces
    print(len(faces))
    if faces is not None:
        for face in faces:
            embedding1 = model(face.unsqueeze(0))  # Get the embedding vector of the face
            print(f"Face embedding: {embedding1}")
            
            # For comparison, you can compare this embedding with embeddings from other known faces
            # Compare with known embeddings for recognition
            # For example, you can store the embeddings of known people and calculate the cosine distance to match
else:
    print("No faces detected.")


embedding2=[]

if box is not None:
    for i in range(box.shape[0]):
        cv2.rectangle(img_rgb2, 
                      (int(box[i][0]), int(box[i][1])), 
                      (int(box[i][2]), int(box[i][3])), 
                      (0, 255, 0), 2)  # Draw bounding box around detected face
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb2)
    plt.axis('off')
    plt.show()
    
    # Extract face embeddings
    faces = mtcnn(img_rgb2)  # Detect faces
    print(len(faces))
    if faces is not None:
        for face in faces:
            embedding2 = model(face.unsqueeze(0))  # Get the embedding vector of the face
            print(f"Face embedding: {embedding2}")
            
            # For comparison, you can compare this embedding with embeddings from other known faces
            # Compare with known embeddings for recognition
            # For example, you can store the embeddings of known people and calculate the cosine distance to match
else:
    print("No faces detected.")
    
    
    
    
    
    
    
# embedding_1 = embedding1[0] / np.linalg.norm(embedding1[0])
# embedding_2 = embedding2[0] / np.linalg.norm(embedding2[0]) 

distance = cosine(embedding1[0].detach().numpy(), embedding2[0].detach().numpy())
print(f"Cosine distance: {distance}")
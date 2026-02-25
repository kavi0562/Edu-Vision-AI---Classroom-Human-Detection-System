import cv2
import os
import numpy as np

def train_recognizer():
    data_dir = "/Users/kavi/human_detection_project/student_faces"
    names = ["Abhijeeth", "Kavishik", "Aryan", "Anand", "Shashank"]
    
    faces = []
    labels = []
    
    print("Loading extracted faces for training...")
    for label, name in enumerate(names):
        img_path = os.path.join(data_dir, f"{name}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found.")
            continue
            
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # We assume the whole image is the face (since we cropped it)
        # But LBPH expects uniform sizes or direct face regions
        faces.append(gray)
        labels.append(label)
        
    print(f"Training LBPH model on {len(faces)} faces...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    # Save the model
    model_path = "/Users/kavi/human_detection_project/face_model.yml"
    recognizer.write(model_path)
    print(f"Model saved to {model_path}")
    print("Label Mapping:")
    for lbl, name in enumerate(names):
        print(f"  {lbl}: {name}")

if __name__ == "__main__":
    train_recognizer()

from deepface import DeepFace
import cv2

try:
    print("Testing DeepFace...")
    # Test on one of the extracted faces
    img_path = "/Users/kavi/human_detection_project/student_faces/Abhijeeth.jpg"
    dfs = DeepFace.find(img_path=img_path, db_path="/Users/kavi/human_detection_project/student_faces", enforce_detection=False, silent=True)
    
    if len(dfs) > 0 and len(dfs[0]) > 0:
        matched = dfs[0].iloc[0]['identity']
        print(f"Match found: {matched}")
    else:
        print("No match found.")
except Exception as e:
    print(f"DeepFace Test Failed: {e}")

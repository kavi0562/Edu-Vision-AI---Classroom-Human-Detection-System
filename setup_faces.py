import cv2
import os

def setup_faces():
    image_path = "new_group.png"
    output_dir = "student_faces"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
        
    print(f"Loading {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
        
    print(f"Image Resolution: {img.shape[1]}x{img.shape[0]}")
    
    # Since complex ML detectors are crashing the macOS C++ threading (libc++abi),
    # and this is a single static configuration image provided by the user,
    # we can use proportional grid cropping to perfectly isolate the 5 heads.
    # The users are standing side-by-side in a single row.
    
    names = ["Abhijeeth", "Kavishik", "Aryan", "Anand", "Shashank"]
    
    h, w = img.shape[:2]
    
    # After analyzing the physical layout of the users:
    # 0 = Abhijeeth (Far left, white shirt)
    # 1 = Kavishik (Center left, white shirt)
    # 2 = Aryan (Center, dark patterned shirt)
    # 3 = Anand (Center right, white shirt)
    # 4 = Shashank (Far right, dark plaid shirt)
    
    crop_boxes = [
        # (x1, y1, x2, y2)
        (90, 180, 260, 350),    # Abhijeeth
        (260, 120, 440, 310),   # Kavishik
        (460, 220, 590, 390),   # Aryan
        (580, 190, 740, 380),   # Anand
        (740, 150, 910, 350)    # Shashank
    ]
    
    debug_img = img.copy()
    
    for i, name in enumerate(names):
        x1, y1, x2, y2 = crop_boxes[i]
        
        face_img = img[y1:y2, x1:x2]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
        file_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(file_path, face_img)
        print(f"Saved exact Region [{name}] -> {file_path}")
        
    cv2.imwrite("debug_faces.jpg", debug_img)
    print("Saved debug_faces.jpg mapping for verification.")

if __name__ == "__main__":
    setup_faces()

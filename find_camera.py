import cv2

def test_cameras():
    print("AI For Her - Camera Finder")
    print("Searching for available camera indices...")
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[FOUND] Camera Index {i} is active and working.")
            cap.release()
        else:
            print(f"[NONE] Camera Index {i} is not available.")

if __name__ == "__main__":
    test_cameras()
    print("\nCheck the list above. Your USB phone is likely Index 1 or 2.")

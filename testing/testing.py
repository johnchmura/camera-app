import cv2
from utilities.leading_lines import detect_leading_lines, draw_detected_lines
from utilities.media_pipe import extract_pose_data

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
pose, lines, circle_center, hitbox = None, None, None, None  # Cached results

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Only compute pose and line detection every 5 frames
    if frame_count % 10 == 0:
        pose = extract_pose_data(frame)  # Extract pose data
        lines, circle_center, hitbox = detect_leading_lines(frame, pose)  # Detect lines

    # Draw lines and bounding box using last computed values
    output_frame = draw_detected_lines(frame, lines, hitbox, circle_center)

    # Display the resulting frame
    cv2.imshow('Video Feed', output_frame)

    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment frame counter

# Release the camera
cap.release()
cv2.destroyAllWindows()

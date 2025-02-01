import cv2
from photography.leading_lines import detect_leading_lines, draw_detected_lines
from photography.rule_thirds import get_rule_thirds
from utilities.media_pipe import extract_pose_data, find_pose, find_pitch, find_roll, find_yaw
from utilities.pose_classifier import make_prediction_data, get_gaze_direction
# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
pose, lines, circle_center, hitbox = None, None, None, None  # Cached results
posture,yaw, roll, pitch,yaw2, roll2, pitch2 = "forward",0,0,0,0,0,0  # Initial angles




while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    frame = cv2.flip(frame, 1)
    # Only compute pose and line detection every 5 frames
    if frame_count % 1 == 0:
        pose = extract_pose_data(frame, running_mode='VIDEO', frame_count=frame_count)
        if pose:
            posture = make_prediction_data(pose)
            roll, yaw, pitch = find_pose(pose)  # Compute head angles
            roll2 = find_roll(pose)* 1000
            yaw2 = find_yaw(pose) * 1000
            pitch2 = find_pitch(pose) * 1000
            lab1 = get_gaze_direction(yaw2)
            #lab2 = get_gaze_direction(yaw2)
            
            
            
        lines, circle_center, hitbox = detect_leading_lines(frame, pose)  # Detect lines
    
    # Draw lines and bounding box using last computed values
    output_frame = draw_detected_lines(frame, lines, hitbox, circle_center)
    output_frame, _ = get_rule_thirds(output_frame, highlight_point=circle_center, posture=posture)
    
    # Display yaw, roll, and pitch in the top-left corner
    cv2.putText(output_frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Roll: {roll:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Pitch: {pitch:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Yaw: {yaw2:f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Roll: {roll2:f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Pitch: {pitch2:f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"label: {posture:s}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video Feed', output_frame)

    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment frame counter

# Release the camera
cap.release()
cv2.destroyAllWindows()

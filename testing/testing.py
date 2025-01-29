from utilities.leading_lines import detect_leading_lines
from utilities.media_pipe import extract_pose_data_from_image

pose = extract_pose_data_from_image("woman_standing1.jpg")
detect_leading_lines("./test_pics/OIP.jpg",pose,True)
detect_leading_lines("./test_pics/download.jpg",pose,True)

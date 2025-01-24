from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt

# Create a detector instance
detector = MTCNN(device="GPU:0")

# Load an image
image = load_image("test.png")

# Detect faces in the image
result = detector.detect_faces(image)


plt.imshow(plot(image, result))
plt.show()

# Display the result
print(result)
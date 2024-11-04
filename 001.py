import cv2
import numpy as np
import os
import time

# Frame dimensions
width = 1920
height = 1080

# Number of circles
number_of_circles = 354

# Create 'video' directory if it doesn't exist
if not os.path.exists('video'):
    os.makedirs('video')

# Get epoch time for filename
epoch_time = int(time.time())

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_filename = os.path.join('video', f'video_{epoch_time}.mp4')
out = cv2.VideoWriter(video_filename, fourcc, 60.0, (width, height))

# Circle class definition
class Circle:
    def __init__(self):
        self.x = np.random.rand() * width
        self.y = np.random.rand() * height
        self.direction = np.random.rand() * 2 * np.pi
        self.r = np.random.randint(0, 256)
        self.g = np.random.randint(0, 256)
        self.b = np.random.randint(0, 256)
        self.a = 0.5  # Not used in this script

# Initialize circles
circles = [Circle() for _ in range(number_of_circles)]

# Main loop
while True:
    # Create a white background image
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Update positions and draw circles
    for i, circle in enumerate(circles):
        # Randomly vary the direction
        circle.direction += (np.random.rand() - 0.5) * 0.1
        # Update position based on direction
        circle.x += np.cos(circle.direction)
        circle.y += np.sin(circle.direction)

        # Bounce off the edges
        if circle.x > width or circle.x < 0 or circle.y > height or circle.y < 0:
            circle.direction += np.pi  # Reverse direction

        # Draw the circle
        cv2.circle(frame, (int(circle.x), int(circle.y)), 5, (0, 0, 0), -1)

    # Interaction between circles
    for i in range(number_of_circles):
        circle_i = circles[i]
        for j in range(i + 1, number_of_circles):
            circle_j = circles[j]
            dx = circle_i.x - circle_j.x
            dy = circle_i.y - circle_j.y
            distance = np.hypot(dx, dy)

            if distance < 10:
                # Adjust directions upon collision
                circle_i.direction += np.pi
                circle_j.direction += np.pi
                circle_i.x += np.cos(circle_i.direction) * 2
                circle_i.y += np.sin(circle_i.direction) * 2
            elif distance < 50:
                # Draw line between close circles
                cv2.line(frame, (int(circle_i.x), int(circle_i.y)), (int(circle_j.x), int(circle_j.y)), (0, 0, 0), 1)

    # Display the frame
    cv2.imshow('Animation', frame)

    # Write the frame to the video file
    out.write(frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()


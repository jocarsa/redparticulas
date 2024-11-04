import cv2
import numpy as np
import os
import time

# Frame dimensions
width = 1920
height = 1080

# Number of particles
number_of_particles = 500

# Create 'video' directory if it doesn't exist
if not os.path.exists('video'):
    os.makedirs('video')

# Get epoch time for filename
epoch_time = int(time.time())

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_filename = os.path.join('video', f'video_{epoch_time}.mp4')
out = cv2.VideoWriter(video_filename, fourcc, 60.0, (width, height))

# Particle class definition
class Particle:
    def __init__(self):
        self.x = np.random.rand() * width
        self.y = np.random.rand() * height
        self.direction = np.random.rand() * 2 * np.pi
        self.r = np.random.randint(0, 256)
        self.g = np.random.randint(0, 256)
        self.b = np.random.randint(0, 256)
        self.a = 0.5  # Not used in this script

# Initialize particles
particles = [Particle() for _ in range(number_of_particles)]

# Main loop
while True:
    # Create a white background image
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Update positions and draw particles
    for particle in particles:
        # Randomly vary the direction
        particle.direction += (np.random.rand() - 0.5) * 0.1
        # Update position based on direction
        particle.x += np.cos(particle.direction)
        particle.y += np.sin(particle.direction)

        # Bounce off the edges
        if particle.x > width or particle.x < 0 or particle.y > height or particle.y < 0:
            particle.direction += np.pi  # Reverse direction

        # Draw the particle with antialiasing
        cv2.circle(
            frame,
            (int(particle.x), int(particle.y)),
            5,
            (0, 0, 0),
            -1,
            lineType=cv2.LINE_AA  # Antialiasing enabled
        )

    # Interaction between particles
    for i in range(number_of_particles):
        particle_i = particles[i]
        for j in range(i + 1, number_of_particles):
            particle_j = particles[j]
            dx = particle_i.x - particle_j.x
            dy = particle_i.y - particle_j.y
            distance = np.hypot(dx, dy)

            if distance < 10:
                # Adjust directions upon collision
                particle_i.direction += np.pi
                particle_j.direction += np.pi
                particle_i.x += np.cos(particle_i.direction) * 2
                particle_i.y += np.sin(particle_i.direction) * 2
            elif distance < 50:
                # Draw line between close particles with antialiasing
                cv2.line(
                    frame,
                    (int(particle_i.x), int(particle_i.y)),
                    (int(particle_j.x), int(particle_j.y)),
                    (0, 0, 0),
                    1,
                    lineType=cv2.LINE_AA  # Antialiasing enabled
                )

    # Display the frame
    cv2.imshow('Particle Network Animation', frame)

    # Write the frame to the video file
    out.write(frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()

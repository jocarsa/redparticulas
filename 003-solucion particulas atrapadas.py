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
        angle = np.random.rand() * 2 * np.pi
        speed = np.random.uniform(1, 3)  # Random speed between 1 and 3
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        self.radius = 5  # Radius of the particle
        self.mass = 1    # Mass of the particle (same for all particles)
        self.color = (0, 0, 0)  # Black color for particles

# Initialize particles
particles = [Particle() for _ in range(number_of_particles)]

# Main loop
while True:
    # Create a white background image
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Update positions
    for particle in particles:
        # Update position based on velocity
        particle.x += particle.vx
        particle.y += particle.vy

        # Bounce off the edges
        if particle.x - particle.radius < 0:
            particle.x = particle.radius
            particle.vx *= -1
        elif particle.x + particle.radius > width:
            particle.x = width - particle.radius
            particle.vx *= -1

        if particle.y - particle.radius < 0:
            particle.y = particle.radius
            particle.vy *= -1
        elif particle.y + particle.radius > height:
            particle.y = height - particle.radius
            particle.vy *= -1

    # Collision detection and response
    for i in range(number_of_particles):
        particle_i = particles[i]
        for j in range(i + 1, number_of_particles):
            particle_j = particles[j]
            dx = particle_j.x - particle_i.x
            dy = particle_j.y - particle_i.y
            distance = np.hypot(dx, dy)
            min_distance = particle_i.radius + particle_j.radius

            if distance < min_distance:
                # Normalize the distance vector
                nx = dx / distance
                ny = dy / distance

                # Relative velocity
                dvx = particle_i.vx - particle_j.vx
                dvy = particle_i.vy - particle_j.vy
                # Relative velocity along the normal
                vn = dvx * nx + dvy * ny

                # If particles are moving towards each other
                if vn > 0:
                    continue

                # Compute impulse scalar
                impulse = (2 * vn) / (particle_i.mass + particle_j.mass)

                # Update velocities
                particle_i.vx -= impulse * particle_j.mass * nx
                particle_i.vy -= impulse * particle_j.mass * ny
                particle_j.vx += impulse * particle_i.mass * nx
                particle_j.vy += impulse * particle_i.mass * ny

                # Position correction to prevent overlap
                overlap = 0.5 * (min_distance - distance + 1)
                particle_i.x -= overlap * nx
                particle_i.y -= overlap * ny
                particle_j.x += overlap * nx
                particle_j.y += overlap * ny

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

    # Draw particles
    for particle in particles:
        cv2.circle(
            frame,
            (int(particle.x), int(particle.y)),
            particle.radius,
            particle.color,
            -1,
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

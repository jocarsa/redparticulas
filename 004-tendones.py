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
                if distance == 0:
                    # Avoid division by zero
                    nx = 0
                    ny = 0
                else:
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
                # Draw tendon-like line between close particles

                # Compute unit direction vector
                if distance == 0:
                    continue  # Skip to avoid division by zero

                dir_x = dx / distance
                dir_y = dy / distance

                # Compute perpendicular vector
                perp_x = -dir_y
                perp_y = dir_x

                # Widths
                end_width = 5  # Thickness at the ends
                center_width = 1  # Thickness at the center

                # Points at particle_i
                p1_x = particle_i.x + perp_x * (end_width / 2)
                p1_y = particle_i.y + perp_y * (end_width / 2)
                p2_x = particle_i.x - perp_x * (end_width / 2)
                p2_y = particle_i.y - perp_y * (end_width / 2)

                # Mid-point
                mid_x = (particle_i.x + particle_j.x) / 2
                mid_y = (particle_i.y + particle_j.y) / 2

                # Points at mid-point
                p3_x = mid_x + perp_x * (center_width / 2)
                p3_y = mid_y + perp_y * (center_width / 2)
                p4_x = mid_x - perp_x * (center_width / 2)
                p4_y = mid_y - perp_y * (center_width / 2)

                # Points at particle_j
                p5_x = particle_j.x + perp_x * (end_width / 2)
                p5_y = particle_j.y + perp_y * (end_width / 2)
                p6_x = particle_j.x - perp_x * (end_width / 2)
                p6_y = particle_j.y - perp_y * (end_width / 2)

                # Construct polygon points in order
                pts = np.array([
                    [p1_x, p1_y],
                    [p3_x, p3_y],
                    [p5_x, p5_y],
                    [p6_x, p6_y],
                    [p4_x, p4_y],
                    [p2_x, p2_y]
                ], np.int32)

                # Draw the filled polygon (tendon)
                cv2.fillPoly(frame, [pts], (0, 0, 0))

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

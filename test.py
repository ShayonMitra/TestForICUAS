#using voronoi points as frontiers this is exploring
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi
import random

# Parameters
SPACE_SIZE = 30
NUM_DRONES = 4
NUM_TAGS = 20
STEPS = 200
DRONE_RANGE = 10
STEP_SIZE = 1.0
REPULSIVE_FORCE_RANGE = 8
REPULSIVE_FORCE_STRENGTH = 15
VISITED_RADIUS = 2

# Randomly spawn ArUco tags
aruco_tags = np.random.uniform(0, SPACE_SIZE, size=(NUM_TAGS, 3))

# Initialize drones
drones = [{"position": np.random.uniform(0, SPACE_SIZE, size=3)} for _ in range(NUM_DRONES)]
visited_points = []
detected_tags = set()
total_reward = 0

def detect_tags(drone_pos, tags, range_limit):
    distances = np.linalg.norm(tags - drone_pos, axis=1)
    detected_indices = np.where(distances <= range_limit)[0]
    return detected_indices

def is_valid_point(point):
    return all(0 <= coord <= SPACE_SIZE for coord in point)

def get_voronoi_regions(points):
    vor = Voronoi(points)
    regions = []
    for r in vor.regions:
        if not -1 in r and len(r) > 0:
            polygon = []
            for i in r:
                v = vor.vertices[i]
                if is_valid_point(v):
                    polygon.append(v)
                else:
                    polygon = None
                    break
            if polygon:
                regions.append(np.array(polygon))
    return regions

def calculate_repulsive_force(drone_pos, other_drones):
    repulsive_force = np.array([0.0, 0.0, 0.0])
    for other_drone in other_drones:
        distance = np.linalg.norm(drone_pos - other_drone)
        if 0 < distance < REPULSIVE_FORCE_RANGE:
            repulsive_force += (drone_pos - other_drone) / distance * (REPULSIVE_FORCE_STRENGTH * (1/distance - 1/REPULSIVE_FORCE_RANGE))
    return repulsive_force

def is_visited(point, visited_points):
    for visited in visited_points:
        if np.linalg.norm(np.array(point) - np.array(visited)) <= VISITED_RADIUS:
            return True
    return False

def find_frontier_voronoi(voronoi, visited_points):
    frontiers = []
    for vertex in voronoi.vertices:
        if is_valid_point(vertex) and not is_visited(vertex, visited_points):
            frontiers.append(vertex)
    return frontiers

# Plotting setup
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, SPACE_SIZE])
ax.set_ylim([0, SPACE_SIZE])
ax.set_zlim([0, SPACE_SIZE])


for step in range(STEPS):
    ax.cla()
    ax.set_xlim([0, SPACE_SIZE])
    ax.set_ylim([0, SPACE_SIZE])
    ax.set_zlim([0, SPACE_SIZE])
    ax.set_title(f"Step: {step + 1}")

    drone_positions = [drone["position"] for drone in drones]
    voronoi = Voronoi(drone_positions)

    for i, drone in enumerate(drones):
        frontiers = find_frontier_voronoi(voronoi, visited_points)

        if frontiers:
            closest_frontier = None
            min_distance = float('inf')
            for frontier in frontiers:
                dist = np.linalg.norm(np.array(frontier) - drone["position"])
                if dist < min_distance:
                    min_distance = dist
                    closest_frontier = frontier
            target_pos = closest_frontier + np.random.uniform(-1, 1, size=3)
            target_pos = np.clip(target_pos, [0, 0, 0], [SPACE_SIZE - 1, SPACE_SIZE - 1, SPACE_SIZE - 1])
        else:
            target_pos = np.random.uniform(0, SPACE_SIZE, size=3)

        # Movement with repulsion
        direction = target_pos - drone["position"]
        other_drones_positions = [d["position"] for j, d in enumerate(drones) if j != i]
        repulsive_force = calculate_repulsive_force(drone["position"], other_drones_positions)
        direction += repulsive_force
        norm = np.linalg.norm(direction)
        if norm > 0:
            drone["position"] += direction / norm * min(norm, STEP_SIZE)
            drone["position"] = np.clip(drone["position"], 0, SPACE_SIZE)

        visited_points.append(drone["position"].copy())

        # Tag detection and plotting
        detected = detect_tags(drone["position"], aruco_tags, DRONE_RANGE)
        newly_detected = [tag for tag in detected if tag not in detected_tags]
        total_reward += len(newly_detected)
        detected_tags.update(detected)

        ax.scatter(*drone["position"], color=f'C{i}', label=f'Drone {i + 1}')

    detected_indices = list(detected_tags)
    if detected_indices:
        ax.scatter(*aruco_tags[detected_indices].T, c='g', label='Detected Tags', s=50)
    undetected_tags = np.delete(aruco_tags, list(detected_tags), axis=0)
    if undetected_tags.size > 0:
        ax.scatter(*undetected_tags.T, c='r', label='Undetected Tags', s=50)

    plt.pause(0.01)

# Print summary
print(f"Simulation complete. Total tags detected: {len(detected_tags)}/{NUM_TAGS}")

plt.show()
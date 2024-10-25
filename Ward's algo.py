import numpy as np
import matplotlib.pyplot as plt

def get_data_points():
    n = int(input("Enter the number of data points: "))
    data = []
    print("Enter the coordinates of each data point (x y):")
    for _ in range(n):
        x, y = map(float, input().split())
        data.append((x, y))
    return np.array(data)

data = get_data_points()
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
n = len(data)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        distance_matrix[i, j] = euclidean_distance(data[i], data[j])
        distance_matrix[j, i] = distance_matrix[i, j]
clusters = [[i] for i in range(n)]
positions = np.arange(n)
def ward_distance(c1, c2):
    combined_cluster = np.vstack((data[c1], data[c2]))
    mean_combined = np.mean(combined_cluster, axis=0)
    variance = np.sum((combined_cluster - mean_combined) ** 2)
    return variance
merge_history = []
heights = []
while len(clusters) > 1:
    min_distance = float('inf')
    clusters_to_merge = (None, None)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = ward_distance(clusters[i], clusters[j])
            if dist < min_distance:
                min_distance = dist
                clusters_to_merge = (i, j)
                print("Distance of: ", clusters[i], clusters[j])
                print(dist, end="\n")
    i, j = clusters_to_merge
    new_cluster = clusters[i] + clusters[j]
    clusters = [clusters[k] for k in range(len(clusters)) if k not in (i, j)]
    clusters.append(new_cluster)
    new_position = (positions[i] + positions[j]) / 2
    positions = np.delete(positions, [i, j])
    positions = np.append(positions, new_position)
    merge_history.append((i, j))
    heights.append(min_distance)
def plot_dendrogram(merge_history, heights):
    plt.figure(figsize=(12, 6))
    current_positions = np.arange(n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(merge_history)))

    for idx, (merge, height) in enumerate(zip(merge_history, heights)):
        i, j = merge
        plt.plot([current_positions[i], current_positions[i]], [0, height], color=colors[idx])
        plt.plot([current_positions[j], current_positions[j]], [0, height], color=colors[idx])
        plt.plot([current_positions[i], current_positions[j]], [height, height], color=colors[idx])

        new_position = (current_positions[i] + current_positions[j]) / 2
        current_positions = np.delete(current_positions, [i, j])
        current_positions = np.append(current_positions, new_position)

    for idx, pos in enumerate(np.arange(n)):
        plt.text(pos, -0.5, f'D{idx+1}', ha='center', va='top', fontsize=10, color='blue', rotation=45)

    plt.title("Enhanced Dendrogram (Manual Implementation)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_dendrogram(merge_history, heights)

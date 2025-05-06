from sklearn.datasets import load_iris
import numpy as np
import random
import matplotlib.pyplot as plt

# Q2: Consider an array like 𝒙 = [4, 3, 6, 7, 5, 13]. Write a Python code to create sub
# array 𝒙_𝒔𝒖𝒃 = [6,7] using slicing.

# The subarray is created by slicing the original array from index 2 to index 4 (exclusive).

x = [4, 3, 6, 7, 5, 13]
x_sub = x[2:4]
print(x_sub)    # [6, 7]



#Q3: To visualize a data point set (x, y) = {(x_i, y_i)}_{i=1}^n with matplotlib:
# Example data points
x = [1, 2, 2.25, 3]  # x-coordinates
y = [1, 1, 3, 0]  # y-coordinates

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Data Point Visualization')
plt.grid(True)
plt.savefig('scatter_plot_q3.png')  # Save the plot as a PNG file

# Assignment 1
# Load dataset
data = load_iris()
X = data.data

def kmeans(X, k=3, max_iters=300, tol=1e-6, verbose=False):
    # Get dimensions of the dataset
    n_samples, n_features = X.shape
    
    # Step 1: Randomly select k data points as initial centroids
    indices = random.sample(range(n_samples), k)
    centroids = X[indices].copy()
    
    # Initialize arrays
    previous_centroids = np.zeros_like(centroids)
    labels = np.zeros(n_samples)
    loss_history = []
    
    # Iterate until convergence or max iterations
    for iteration in range(max_iters):
        # Step 2 & 3: Calculate distances and assign clusters
        distances = np.zeros((n_samples, k))
        for i in range(k):
            # Calculate Euclidean distance for each centroid
            sq_diff = (X - centroids[i])**2
            distances[:, i] = np.sum(sq_diff, axis=1)
        
        # Assign each point to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Calculate current loss (sum of squared distances to assigned centroids)
        current_loss = 0
        for i in range(n_samples):
            current_loss += distances[i, int(labels[i])]
        loss_history.append(current_loss)
        
        if verbose:
            print(f"Iteration {iteration}, Loss: {current_loss}")
        
        # Save current centroids for convergence check
        previous_centroids = centroids.copy()
        
        # Step 4: Update centroids based on assigned points
        for i in range(k):
            # Get all points assigned to this cluster
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        
        # Step 5: Check for convergence - if centroids barely moved
        centroid_change = np.sum((centroids - previous_centroids)**2)
        if centroid_change < tol:
            if verbose:
                print(f"Converged at iteration {iteration} with change {centroid_change}")
            break
    
    return centroids, labels, np.array(loss_history)

# Step 6: Run K-means 10 times with different initializations
best_loss = float('inf')
best_centroids = None
best_labels = None
best_loss_history = None

for run in range(10):
    print(f"K-means run {run+1}/10")
    centroids, labels, loss_history = kmeans(X, k=3, verbose=True)
    final_loss = loss_history[-1]
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_centroids = centroids
        best_labels = labels
        best_loss_history = loss_history
        print(f"New best loss: {best_loss}")

print("\nBest Centroids:")
print(best_centroids)
print("\nbest_labels", best_labels)
print(f"\nSum of Squared Errors (SSE): {best_loss}")

# Step 7: Plot the convergence (loss vs iterations)
plt.figure(figsize=(10, 6))
plt.plot(range(len(best_loss_history)), best_loss_history, 'o-')
plt.title('K-means Convergence Plot (Best Run)')
plt.xlabel('Iteration')
plt.ylabel('Loss (Sum of Squared Distances)')
plt.grid(True)
plt.tight_layout()
plt.savefig('kmeans_convergence.png')

# Visualize the final clustering result (using first two features)
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X[best_labels == i, 0], X[best_labels == i, 1], 
                c=colors[i], label=f'Cluster {i+1}')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], marker='*', 
            s=300, c='black', label='Centroids')
plt.title('K-means Clustering Result (Best Run)')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_clusters.png')

print("\nBest K-means run had final loss:", best_loss)
print("Convergence plot saved as 'kmeans_convergence.png'")
print("Clustering result saved as 'kmeans_clusters.png'")
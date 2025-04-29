import matplotlib.pyplot as plt
import numpy as np

# Given data as a list of tuples
data = [(1, 2), (3, 1), (3, 3), (4, 2)]

# Create the x and y arrays
x = [point[0] for point in data]
y = [point[1] for point in data]

# Convert x and y to numpy arrays for easier calculations
x = np.array(x)
y = np.array(y)

# Hypothesis functions for Model A and Model B
def model_a(x, theta):
    return theta[0] + theta[1] * x

def model_b(x, theta):
    return theta[0] + theta[1] * x

# Parameters for Model A and Model B
theta_a = np.array([0, 1])
theta_b = np.array([2, 0])

# Generate predictions for Model A and Model B
y_pred_a = model_a(x, theta_a)
y_pred_b = model_b(x, theta_b)

# Calculate Mean Squared Error (MSE) Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

loss_a = mse_loss(y, y_pred_a)
loss_b = mse_loss(y, y_pred_b)

# Plot the data and hypothesis curves
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', marker='o', s=100, label='Dataset')
plt.plot(x, y_pred_a, color='red', label='Model A')
plt.plot(x, y_pred_b, color='green', label='Model B')
plt.title('Dataset and Hypothesis Curves')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.xticks(range(min(x)-1, max(x)+2))
plt.yticks(range(min(y)-1, max(y)+2))
plt.legend()

# Check if an interactive backend is available, if not, save to a file.
if plt.isinteractive():
    plt.show()  # Display the plot
else:
    plt.savefig('documents/cartesian_plot.png') #save the plot
    print("Plot saved to 'documents/cartesian_plot.png'")

# Print the loss values
print(f"Loss for Model A: {loss_a}")
print(f"Loss for Model B: {loss_b}")
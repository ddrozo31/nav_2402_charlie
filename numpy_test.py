import numpy as np
import matplotlib.pyplot as plt

# Define the size of the matrix
rows = 200
cols = 200

# Define the positions of the obstacles
obstacles = [(1, 1), (1, 2), (1, 3)]

# Initialize the matrix with 0s
matrix = np.zeros((rows, cols), dtype=int)

ones = np.ones((rows, cols), dtype=int)

# Place the obstacles
for c in range(cols):
    matrix[0][c] = 1

for c in range(cols):
    matrix[rows-1][c] = 1

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

# Plot obstacles and blank spaces
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == 1:
            ax.plot(j, i, 's', color='black', markersize=1)  # Mark obstacles with black squares
        else:
            ax.plot(j, i, 's', color='white', markersize=1)  # Mark blank spaces with white squares and black border

# Set limits, grid and aspect ratio
#ax.set_xlim(-10.0, cols - 10.0)
#ax.set_ylim(rows - 10.0, -10.0)
#ax.set_xticks(np.arange(cols))
#ax.set_yticks(np.arange(rows))
ax.grid(True)
ax.set_aspect('equal')

# Annotate each cell with the numeric value (0 or 1)
#for i in range(rows):
#    for j in range(cols):
#        ax.text(j, i, str(matrix[i, j]), va='center', ha='center', color='red')

# Hide the major ticks
#ax.tick_params(which='major', size=0)

# Show the plot
plt.show()
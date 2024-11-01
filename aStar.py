import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import random
import matplotlib.animation as animation

maze_size = 51

# Tạo ma trận 
def generate_maze(size):
    maze = np.ones((size, size), dtype=int)  # Bắt đầu tạo tường
    stack = [(1, 1)]
    maze[1, 1] = 0 #Điểm bắt đầu

    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    while stack:
        current = stack[-1]
        x, y = current
        
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                neighbors.append((nx, ny))
        
        if neighbors:

            nx, ny = random.choice(neighbors)
            maze[(x + nx) // 2, (y + ny) // 2] = 0  
            maze[nx, ny] = 0 
            stack.append((nx, ny))  
        else:
            stack.pop() 

 
    maze[1, 1] = 0  
    maze[size - 2, size - 2] = 0  
    return maze

start = (1, 1)
goal = (maze_size - 2, maze_size - 2)

#Sử dụng hàm ước tính chi phí
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    rows, cols = maze.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    visited = set()
    
    # SetUp điểm
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap='binary')
    ax.scatter(start[1], start[0], color="green", s=100, label="Start")
    ax.scatter(goal[1], goal[0], color="red", s=100, label="Goal")
    path_plot, = ax.plot([], [], color="blue", linewidth=5)
    ax.legend()
    
    def update_plot():
        if goal in came_from:
            path = []
            current = goal
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            x_coords, y_coords = zip(*path[::-1])
            path_plot.set_data(y_coords, x_coords)
        plt.pause(0.00001) 

    while not open_set.empty():
        _, current = open_set.get()
        
        if current == goal:
            update_plot()
            plt.title("Path Found")
            plt.show()
            return 
        
        if current in visited:
            continue
        visited.add(current)
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[neighbor[0]][neighbor[1]] == 1 or neighbor in visited:
                    continue  # Skip walls and visited nodes
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))
                    
                  
                    ax.plot(neighbor[1], neighbor[0], "o", color="yellow", markersize=2)
        
        update_plot()

    plt.title("No Path Found")
    plt.show()

maze = generate_maze(maze_size)
a_star(maze, start, goal)

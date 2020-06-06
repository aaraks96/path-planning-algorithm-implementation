# path-planning-algorithm-implementation
Modules to import: numpy, cv2, math

Files to run: 
For point robot: Proj_2_astar_point.py, Proj_2_djikstra_point.py

It prompts to enter start node x position, start node y position, goal node x position and goal node y position. These inputs must be inside the limits of the grid (0,0) to (250,150). If it goes below or above the limit, it automatically sets it to the lower or upper value. If the nodes given as input lie inside the obstacle, the code exits after displaying a message. Output will show the obstacles and the node exploration.

For rigid robot: Proj_2_astar.py, Proj_2_djikstra.py

It prompts to enter start node x position, start node y position, goal node x position, goal node y position, robot dimension and robot clearance. These inputs must be inside the limits of the grid (0,0) to (250,150). If it goes below or above the limit, it automatically sets it to the lower or upper value. If the nodes given as input lie inside the obstacle, the code exits after displaying a message. Output will show the obstacles, modified obstacle space and the node exploration.

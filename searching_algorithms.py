from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Breadth-First Search (BFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None: 
        return False
    queue = deque()
    queue.append(start)
    visited = {start}
    came_from = {}
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current = queue.popleft()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                grid.draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()
        grid.draw()
        if current != start:
            current.make_closed()

    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None: 
        return False
    
    stack = [start]
    visited = {start}
    came_from = {}
    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                grid.draw()
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()
        grid.draw()
        if current != start:
            current.make_closed()

    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

from math import sqrt
def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

from queue import PriorityQueue
def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None: 
        return False
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    closed_list = [[False for _ in range (grid.cols)] for _ in range (grid.rows)]
    came_from = {}

    g_score = {}
    for row in grid.grid:
        for spot in row:
            g_score[spot] = float("inf")
    g_score[start] = 0

    f_score = {}
    for row in grid.grid:
        for spot in row:
            f_score[spot] = float("inf")
    f_score[start] = h_manhattan_distance(start.get_position(), end.get_position())

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current_f, current = open_set.get()

        if closed_list[current.row][current.col]:
            continue
        closed_list[current.row][current.col] = True
        
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                grid.draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            if closed_list[neighbor.row][neighbor.col]:
                continue

            pot_g = g_score[current] + 1
            if pot_g < g_score.get(neighbor):
                came_from[neighbor] = current
                g_score[neighbor] = pot_g
                f_score[neighbor] = pot_g + h_manhattan_distance(neighbor.get_position(), end.get_position())
                open_set.put((f_score[neighbor],neighbor))
                neighbor.make_open()
        grid.draw()
        if current != start:
            current.make_closed()
    return False

def dls(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depth-Limited Search (DLS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None: 
        return False
    
    stack = [(start, 0)]
    visited = {start}
    came_from = {}
    limit = 30
    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current, depth = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                grid.draw()
            end.make_end()
            return True
        if depth == limit:
            continue
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append((neighbor, depth + 1))
                neighbor.make_open()
        grid.draw()
        if current != start:
            current.make_closed()

    return False

def dijkstra(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Dijkstra Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None: 
        return False
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    closed_list = [[False for _ in range (grid.cols)] for _ in range (grid.rows)]
    came_from = {}

    g_score = {}
    for row in grid.grid:
        for spot in row:
            g_score[spot] = float("inf")
    g_score[start] = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current_g, current = open_set.get()

        if closed_list[current.row][current.col]:
            continue
        closed_list[current.row][current.col] = True
        
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                grid.draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            if closed_list[neighbor.row][neighbor.col]:
                continue

            pot_g = g_score[current] + 1
            if pot_g < g_score.get(neighbor):
                came_from[neighbor] = current
                g_score[neighbor] = pot_g
                open_set.put((g_score[neighbor],neighbor))
                neighbor.make_open()
        grid.draw()
        if current != start:
            current.make_closed()
    return False

# What is the difference between this and Dijkstra?
def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Uniform cost search Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None: 
        return False
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    closed_list = [[False for _ in range (grid.cols)] for _ in range (grid.rows)]
    came_from = {}

    g_score = {}
    for row in grid.grid:
        for spot in row:
            g_score[spot] = float("inf")
    g_score[start] = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current_g, current = open_set.get()

        if closed_list[current.row][current.col]:
            continue
        closed_list[current.row][current.col] = True
        
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                grid.draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            if closed_list[neighbor.row][neighbor.col]:
                continue

            pot_g = g_score[current] + 1
            if pot_g < g_score.get(neighbor):
                came_from[neighbor] = current
                g_score[neighbor] = pot_g
                open_set.put((g_score[neighbor],neighbor))
                neighbor.make_open()
        grid.draw()
        if current != start:
            current.make_closed()
    return False

def iddfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None: 
            return False
    
    def dls_exec(draw: callable, grid: Grid, start: Spot, end: Spot, limit: int) -> bool:
        
        stack = [(start, 0)]
        visited = {start}
        came_from = {}
        while stack:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
            current, depth = stack.pop()
            if current == end:
                while current in came_from:
                    current = came_from[current]
                    current.make_path()
                    grid.draw()
                end.make_end()
                return True
            if depth == limit:
                continue
            for neighbor in current.neighbors:
                if neighbor not in visited and not neighbor.is_barrier():
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    neighbor.make_open()
            grid.draw()
            if current != start:
                current.make_closed()
        return False
    
    max_depth = grid.rows * grid.cols

    for limit in range(1, max_depth):
        for row in grid.grid:
            for spot in row:
                if not spot.is_barrier() and not spot.is_start() and not spot.is_end():
                    spot.reset()
        
        if dls_exec(draw, grid, start, end, limit):
            return True
    return False


# and the others algorithms...
# ▢ Uninformed Cost Search (UCS)
# ▢ Iterative Deepening Search/Iterative Deepening Depth-First Search (IDS/IDDFS)
# ▢ Iterative Deepening A* (IDA)
# Assume that each edge (graph weight) equalss
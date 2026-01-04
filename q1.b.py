import numpy as np
import copy
import heapq
from scipy.optimize import linear_sum_assignment
from collections import deque
# --- GLOBAL SETTINGS ---
PRECOMPUTED = {}
SIZE = 6

# Knight moves (row, col) - All 8 possible "L" shapes
MOVES = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

# This will hold the map layout (walls/swamps) so we don't lose them when horses move
ORIGINAL_MAP = []


# --- 1. BFS & PRECOMPUTATION ---

def bfs_from_start(start_r, start_c):
    """
    Runs a BFS from a specific square to ALL other squares on the board.
    Returns a 2D array of distances (steps) ignoring other horses.
    """
    # Initialize distances matrix with infinity
    dists = [[float('inf')] * SIZE for _ in range(SIZE)]
    dists[start_r][start_c] = 0
    queue = deque([(start_r, start_c)])

    while queue:
        r, c = queue.popleft()
        d = dists[r][c]

        for dr, dc in MOVES:
            tr, tc = r + dr, c + dc
            # Check boundaries
            if 0 <= tr < SIZE and 0 <= tc < SIZE:
                # If we haven't visited this cell yet
                if dists[tr][tc] == float('inf'):
                    dists[tr][tc] = d + 1
                    queue.append((tr, tc))
    return dists


def precompute():
    """
    Calculates and caches the shortest path (BFS distance) between
    ANY two points on the board. This makes the heuristic calculation very fast.
    """
    global PRECOMPUTED
    if PRECOMPUTED:
        return

    # Iterate over every possible start square
    for r1 in range(SIZE):
        for c1 in range(SIZE):
            table = bfs_from_start(r1, c1)
            # Store distance to every possible end square
            for r2 in range(SIZE):
                for c2 in range(SIZE):
                    # Key: ((start_r, start_c), (end_r, end_c)) -> Value: Distance
                    PRECOMPUTED[((r1, c1), (r2, c2))] = table[r2][c2]


# --- 2. HEURISTIC ---

def get_horses(board):
    """ Helper: Returns a list of (row, col) tuples for all horses (value 3) """
    horses = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 3:
                horses.append((r, c))
    return horses


def heuristic(board, goals):
    """
    Calculates the 'H' value using the Hungarian Algorithm.
    It finds the optimal assignment of horses to goals based on BFS distances.
    """
    horses = get_horses(board)
    n = len(horses)

    if n == 0: return 0
    if n != len(goals): return float('inf')

    # Build cost matrix: Rows = Horses, Cols = Goals
    matrix = np.zeros((n, n))
    for i, h in enumerate(horses):
        for j, g in enumerate(goals):
            # Fetch precomputed BFS distance
            dist = PRECOMPUTED.get((h, g), float('inf'))
            matrix[i, j] = dist

    # Solve assignment problem (min cost perfect matching)
    row, col = linear_sum_assignment(matrix)

    # Return the sum of distances for the optimal matching
    return matrix[row, col].sum()


# --- 3. MOVES & LOGIC ---

def get_cost(r, c):
    """ Returns the cost of moving FROM (r, c). 2 if Swamp, 1 otherwise. """
    # Check if we are leaving a swamp (2).
    # We look at ORIGINAL_MAP because the board has a horse (3) there now.
    if ORIGINAL_MAP[r][c] == 2:
        return 2
    return 1


def apply_move(board, move):
    """ Generates a NEW board object after a specific move. """
    # Create new board state
    new_b = copy.deepcopy(board)
    r1, c1, r2, c2 = move

    # 1. Remove horse from old spot
    # Restore what was there originally (0 or 2) using the static map
    if ORIGINAL_MAP[r1][c1] == 2:
        new_b[r1][c1] = 2
    else:
        new_b[r1][c1] = 0

    # 2. Place horse in new spot
    new_b[r2][c2] = 3
    return new_b


def get_legal_moves(board):
    """ Returns a list of all valid [r1, c1, r2, c2] moves for current board. """
    horses = get_horses(board)
    moves = []

    for h in horses:
        r, c = h
        for dr, dc in MOVES:
            tr, tc = r + dr, c + dc

            # Check bounds
            if 0 <= tr < SIZE and 0 <= tc < SIZE:
                val = board[tr][tc]
                # Can land on Empty (0) or Swamp (2)
                # Cannot land on Wall (1), Another Horse (3), or Water (4)
                if val == 0 or val == 2:
                    moves.append([r, c, tr, tc])
    return moves


# --- 4. A* SEARCH ---

def get_hash(board):
    """ Converts list-of-lists board to a tuple-of-tuples for dictionary keys. """
    return tuple(map(tuple, board))


def is_goal(b1, b2):
    return b1 == b2


def init_A_star(start, goals):
    """ Initializes the priority queue and dictionaries for A*. """
    h = heuristic(start, goals)

    parents = {}  # Maps: Child_Hash -> Parent_Hash
    boards_db = {}  # Maps: Hash -> Actual Board Object (list)

    start_hash = get_hash(start)
    parents[start_hash] = None
    boards_db[start_hash] = start

    g_scores = {start_hash: 0}  # Maps: Hash -> Cost so far

    # Priority Queue stores: (f_score, g_score, board_object)
    pq = []
    heapq.heappush(pq, (h, 0, start))

    return pq, g_scores, parents, boards_db


def reconstruct(end_hash, parents, boards_db):
    """ Rebuilds the path from Goal back to Start using the parents map. """
    path = []
    curr = end_hash
    while curr is not None:
        # Retrieve the full board object from the DB using the hash
        path.append(boards_db[curr])
        # Move to the parent
        curr = parents.get(curr)
    path.reverse()  # Reverse the list so it goes Start -> Goal
    return path


def expand(g, board, b_hash, pq, g_scores, parents, boards_db, goals):
    """ Expands the current node: Generates children and updates A* structures. """
    moves = get_legal_moves(board)

    for m in moves:
        r, c = m[0], m[1]

        # Calculate cost based on  (leaving swamp = 2, else 1)
        cost = get_cost(r, c)

        # Create the new state
        child_board = apply_move(board, m)
        child_hash = get_hash(child_board)

        new_g = g + cost

        # A* Logic: If found a new path OR a shorter path to this state
        if new_g < g_scores.get(child_hash, float('inf')):
            # 1. Update G-score
            g_scores[child_hash] = new_g
            # 2. Update Parent pointer
            parents[child_hash] = b_hash
            # 3. Store the actual board for later retrieval
            boards_db[child_hash] = child_board

            # 4. Calculate F = G + H
            h = heuristic(child_board, goals)
            f = new_g + h

            # 5. Push to priority queue
            heapq.heappush(pq, (f, new_g, child_board))


def find_path(start_board, goal_board, search_method, detail_output):
    """ Main A* Loop. """
    global ORIGINAL_MAP

    # Vars for capturing 2nd node heuristic
    second_node_h = None
    first_pop = True

    if search_method != 1:
        return "Not implemented", 0, 0

    precompute()
    goals = get_horses(goal_board)

    # Save the static map layout to handle swamps correctly
    ORIGINAL_MAP = copy.deepcopy(start_board)
    pq, g_scores, parents, boards_db = init_A_star(start_board, goals)

    while pq:
        # Pop the node with the lowest F score
        f, g, curr_board = heapq.heappop(pq)
        curr_hash = get_hash(curr_board)

        # Catch H for the second node
        if g > 0 and first_pop:
            second_node_h = f - g  # Since f = g + h, then h = f - g
            first_pop = False

        # Lazy Deletion: Skip if we found a better path to this node already
        if g > g_scores.get(curr_hash, float('inf')):
            continue

        # Check for goal
        if is_goal(curr_board, goal_board):
            path = reconstruct(curr_hash, parents, boards_db)
            if not detail_output:
                second_node_h = None
            # Return path, final cost, and H value
            return path, g, second_node_h

        # Expand neighbors
        expand(g, curr_board, curr_hash, pq, g_scores, parents, boards_db, goals)

    return "No path found", 0, 0


# --- 5. PRINTING ---

def print_board(board, title=""):
    """ Utility to print the board nicely with symbols. """
    chars = {0: " ", 1: "@", 2: "&", 3: "*", 4: "="}

    if title:
        print(f"\n{title}:")
        print()

    print("   1  2  3  4  5  6")
    for i in range(len(board)):
        row_str = f"{i + 1}: "
        vals = [chars.get(x, '?') for x in board[i]]
        row_str += "  ".join(vals)
        print(row_str)
    print("-----")


def show_solution(path, cost, h_val):
    # 1. Validation: Check if the algorithm returned an error message (str) instead of a list
    if not isinstance(path, list):
        print(path)  # Print "No path found" or similar message
        return

    # 2. Main Loop: Iterate through every board state ("snapshot") in the solution path
    # enumerate provides both the index (i) and the board object (b)
    for i, b in enumerate(path):

        # 3. Labeling: Name the first board "Start", others "Step X"
        name = "Start" if i == 0 else f"Step {i}"

        # 4. Visualization: Call the function that actually draws the board on screen
        print_board(b, name)

        # 5. Bonus Requirement: Check if we are at the second node (step 1)
        # AND verify that we actually captured a heuristic value
        if i == 1 and h_val is not None:
            print(f"Heuristic of 2nd node: {h_val:.2f}")
            print("-----")

    # 6. Summary: Print total steps and final cost at the end
    print(f"Total Steps: {len(path) - 1}")  # Subtract 1 because 'Start' isn't a step
    print(f"Final Weighted Cost: {cost}")
    print("--- END ---")


# --- MAIN EXECUTION (TRAP TEST) ---

# This setup forces the horse to step on a Swamp (2)
OVER_SWAMP_START = [
    [3, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

OVER_SAMP_GOAL = [
    [0, 0, 0, 0, 3, 0],  # Goal at (0,4)
    [0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

STARTING_BOARD = [
    [3, 0, 3, 0, 2, 0],  # * &
    [0, 0, 0, 3, 1, 4],  # * @ =
    [1, 0, 2, 0, 0, 0],  # &
    [0, 0, 1, 0, 1, 0],  # @   @
    [2, 0, 0, 4, 0, 4],  # = =
    [0, 1, 0, 2, 0, 0]  # @   &
]

GOAL_BOARD = [
    [0, 0, 0, 0, 2, 3],  # & *
    [0, 0, 0, 0, 1, 4],  # @ =
    [1, 0, 2, 0, 0, 0],  # @ &
    [0, 0, 1, 0, 1, 3],  # @ @ *
    [2, 0, 3, 4, 0, 4],  # & * = =
    [0, 1, 0, 2, 0, 0]  # @ &
]





STARTING_BOARD_TRAP = [
    #  1  2  3  4  5  6
    [3, 0, 0, 1, 0, 0],  # H1(0,0) - נגיש רק ל-R2
    [0, 1, 1, 1, 1, 0],  # @ @ @
    [0, 1, 3, 0, 0, 0],  # H2(2,2) - נגיש רק ל-R4
    [1, 1, 1, 1, 1, 1],  # מחסום אופקי מלא!
    [0, 1, 0, 1, 4, 0],  # RIDER G1 (4,4)
    [0, 0, 0, 0, 0, 4]   # RIDER G2 (5,5)
]

GOAL_BOARD_TRAP = [
    #  1  2  3  4  5  6
    [0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 3, 1, 4, 0], # 🎯 H1 ב-(4,2) ליד R1
    [0, 0, 3, 0, 0, 4]  # 🎯 H2 ב-(5,3) ליד R2
]


start = [
    [0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
goal = [
    [0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0]
]

path, cost, h_val = find_path(start, goal, 1, True)
show_solution(path, cost, h_val)
import numpy as np  # Used for matrix operations (needed for the Hungarian Algorithm cost matrix)
import copy  # Used to create deep copies of the board (so moves don't affect the original state)
import heapq  # Used for the Priority Queue implementation in A* (efficiently fetching min f-score)
import random
import math
from scipy.optimize import \
    linear_sum_assignment  # The specific function that solves the Assignment Problem (Hungarian Algo)
from collections import deque  # Used for the BFS implementation (queue for FIFO)

# ... [The rest of the global variables and helper functions remain exactly the same] ...
# ... [No changes to logic or A* functions] ...

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
    try:
        # Solve assignment problem (min cost perfect matching)
        row, col = linear_sum_assignment(matrix)

        # Return the sum of distances for the optimal matching
        return matrix[row, col].sum()
    except ValueError:
        return float('inf')


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




def find_path(starting_board, goal_board, search_method, detail_output):
    """
    Main Dispatcher Function.
    This function identifies the requested search method and calls the
    corresponding solver function.
    """
    precompute()
    # Method 1: A* Search
    if search_method == 1:
        solve_a_star(starting_board, goal_board, detail_output)

    # Method 2: Hill Climbing
    elif search_method == 2:
        solve_hill_climbing(starting_board, goal_board)

    # Method 3: Simulated Annealing (Placeholder for future implementation)
    elif search_method == 3:
        solve_simulated_annealing(starting_board,goal_board,detail_output)


    # Method 4: Local Beam Search (Placeholder)
    elif search_method == 4:
        solve_local_beam(starting_board,goal_board,detail_output)


    # Method 5: Genetic Algorithm (Placeholder)
    elif search_method == 5:
        solve_genetic(starting_board,goal_board,detail_output)



# ------------------------------------------------------------------
# Algorithm Implementations
# ------------------------------------------------------------------

def solve_a_star(start_board, goal_board, detail_output):
    """
    Implementation of A* Algorithm (Method 1).
    Logic preserved from original snippet.
    """
    global ORIGINAL_MAP
    found_solution = False
    # Variables for capturing the heuristic of the 2nd node (for detail_output)
    second_node_h = None
    first_pop = True

    # Pre-computation and setup
    goals = get_horses(goal_board)

    # Save the static map layout to handle swamps/walls correctly
    ORIGINAL_MAP = copy.deepcopy(start_board)

    # Initialize A* structures (Priority Queue, Scores, Parents, Board DB)
    pq, g_scores, parents, boards_db = init_A_star(start_board, goals)

    while pq:
        # Pop the node with the lowest F score (f = g + h)
        f, g, curr_board = heapq.heappop(pq)
        curr_hash = get_hash(curr_board)

        # Logic to capture the H value of the first node chosen (for detailed output)
        # Since f = g + h, we can derive h = f - g
        if g > 0 and first_pop:
            second_node_h = f - g
            first_pop = False

        # Lazy Deletion: Skip if we found a better path to this node already
        if g > g_scores.get(curr_hash, float('inf')):
            continue

        # Check if the current board matches the goal configuration
        if is_goal(curr_board, goal_board):
            path = reconstruct(curr_hash, parents, boards_db)

            # If detail output is not required, we don't need to return the h value
            if not detail_output:
                second_node_h = None

            # Return path, final cost (g), and the heuristic of the first step
            show_solution(path,1,second_node_h,None,detail_output)
            found_solution = True
            break

        # Expand neighbors and add valid moves to the priority queue
        expand(g, curr_board, curr_hash, pq, g_scores, parents, boards_db, goals)
    if not found_solution:
        show_solution("No path found", 1,None,None,detail_output)






def solve_hill_climbing(start_board, goal_board):
    """
    Main Manager for Hill Climbing.
    Controls the restarts and the 'already_checked' list.
    """
    global ORIGINAL_MAP
    ORIGINAL_MAP = copy.deepcopy(start_board)

    # 1. Setup goals and the ban list
    goals = get_horses(goal_board)
    already_checked = []  # Stores starting moves that led to failure

    # 2. Restart Loop (Try up to 5 times)
    for attempt in range(1, 6):
        # 3. Run a single climbing attempt
        success, start_move_used = run_climbing_attempt(start_board, goals, already_checked)

        if success:
            show_solution(success, 2, None, None, False)
            return
        # 4. If failed, ban the starting move of this attempt
        if start_move_used:
            already_checked.append(start_move_used)

    # 5. If loop finishes without success
    show_solution("No path found", 2, None, None, False)


def run_climbing_attempt(start_board, goals, already_checked):
    """
    Performs one single Hill Climbing run.
    Returns (True, None) if solved, or (False, start_move) if stuck.
    """
    path = [copy.deepcopy(start_board)]

    # A. Find the first move (respecting the already_checked list)
    h_start, start_move = find_best_move(start_board, already_checked, goals)

    # If no valid moves exist from start, fail immediately
    if h_start == -1:
        return False, None

    # Apply the first move
    current_board = apply_move(start_board, start_move)
    path.append(copy.deepcopy(current_board))

    # Check immediate win
    if h_start == 0:

        return path, None

    # B. The Main Climbing Loop
    while True:
        # Find best move from CURRENT state
        h_now, move = find_best_move(current_board, [], goals)

        # Apply the move
        current_board = apply_move(current_board, move)
        path.append(copy.deepcopy(current_board))

        # Check win
        if h_now == 0:

            return path, None

        # Look ahead to check for Local Optimum
        h_new, next_move = find_best_move(current_board, already_checked, goals)

        # C. Stop condition: If we are not improving (Local Optimum)
        if h_now < h_new:
            # We got stuck. Return False and the move that started this bad path
            return False, start_move


def find_best_move(board, already_checked, goals):
    """
    Scans all legal moves and returns the one with the lowest heuristic.
    Returns (best_h, best_move) or (-1, None) if no moves.
    """
    moves = get_legal_moves(board)
    chosen_move = None
    h_chosen = float('inf')

    for m in moves:
        # Only filter moves if they are in the 'already_checked' ban list
        if m not in already_checked:
            board_temp = apply_move(board, m)
            h = heuristic(board_temp, goals)

            if h < h_chosen:
                h_chosen = h
                chosen_move = m

    if chosen_move is None:
        return -1, None

    return h_chosen, chosen_move




def solve_simulated_annealing(starting_board, goal_board, detail_output):
    """
    Main Manager for Simulated Annealing.
    Fixed: Now correctly updates prob_to_print and h_start for the first step.
    """
    global ORIGINAL_MAP
    ORIGINAL_MAP = copy.deepcopy(starting_board)

    # 1. Initialization
    goals = get_horses(goal_board)
    current_board = copy.deepcopy(starting_board)
    path = [copy.deepcopy(starting_board)]

    h_now = heuristic(current_board, goals)
    first_move_chosen = None
    T_start = 10.0
    moves_to_print = []
    # 2. Main Loop
    for t in range(0, 100):

        # Calculate Temperature
        T = T_start * ((100 - t) / 100) ** 2
        if T < 0.0001: T = 0.0001

        # 3. Perform Logic Step
        # Now returns the CANDIDATE data regardless of acceptance
        step_res = perform_annealing_step(current_board, h_now, goals, T,moves_to_print,first_move_chosen)
        is_win, move_accepted, candidate_board, candidate_h, probability,moves_to_print = step_res
        # Handle Immediate Win
        if is_win:
            path.append(copy.deepcopy(candidate_board))
            show_solution(path, 3, first_move_chosen, moves_to_print, detail_output)
            return

        # Handle Stuck Case (No moves)
        if candidate_board is None:
            break

        # Handle Move Acceptance
        if move_accepted:
            current_board = candidate_board
            h_now = candidate_h
            path.append(copy.deepcopy(current_board))
            # 4. FIRST STEP LOGIC
            if first_move_chosen is None:
                first_move_chosen = moves_to_print[-1][0]

    # 5. End of loop
    show_solution("No path found", 3, None, None, detail_output)


def perform_annealing_step(board, h_now, goals, T, moves_to_print,first_move_chosen):
    """
    Worker function: Calculates probability and decides on the move.
    Returns: (is_win, move_accepted, new_board, new_h, probability)
    """
    moves = get_legal_moves(board)
    if len(moves) == 0:
        return False, False, None, h_now, 0.0,moves_to_print

    # Pick a random neighbor
    move_new = random.choice(moves)
    check_board = apply_move(board, move_new)
    h_new = heuristic(check_board, goals)

    # Check for immediate win
    if h_new == 0:
        if first_move_chosen is None:
            moves_to_print.append((move_new,h_new,1))
        return True, True, check_board, 0, 1.0,moves_to_print

    # Calculate Energy Delta
    # delta > 0 means improvement (new H is smaller)
    delta_E = h_now - h_new

    # CASE 1: Improvement (Good move)
    # We always accept improvements. Probability is effectively 1.0.
    if delta_E > 0:
        if first_move_chosen is None:
            moves_to_print.append((move_new,h_new,1))
        return False, True, check_board, h_new, 1.0,moves_to_print

    # CASE 2: Bad move (Worsening or same state)
    else:
        # Calculate probability using Metropolis formula
        try:
            probability = math.exp(delta_E / T)
        except OverflowError:
            probability = 1.0
        if first_move_chosen is None:
            moves_to_print.append((move_new,h_new,probability))

        # Determine if we accept this bad move based on the probability
        if random.random() < probability:
            # Accepted bad move
            return False, True, check_board, h_new, probability,moves_to_print
        else:
            # Rejected bad move
            return False, False, check_board, h_new, probability,moves_to_print






def solve_local_beam(starting_board, goal_board, detail_output):
    """
    Main Manager for Local Beam Search.
    Coordinates the loop, expansion, and selection process.
    """
    global ORIGINAL_MAP
    ORIGINAL_MAP = copy.deepcopy(starting_board)

    # 1. Initialization
    # Define the beam width (k)
    k = 3
    goals = get_horses(goal_board)
    bag_first_move = []
    # Initialize the first state: (Heuristic Score, Board, Path History)
    start_h = heuristic(starting_board, goals)
    first_state = (start_h, starting_board, [starting_board],None)

    # The 'current_states' list holds our k best boards
    current_states = [first_state]

    # 2. Main loop (Safety limit of 100 steps)
    for t in range(100):

        # Step A: Expand all current states to find all possible children
        # Returns either a winning path (if found) or a list of candidates
        solution_path,all_candidates = generate_all_candidates(current_states, goals)
        if t==0:
            for h,b,p,m in all_candidates:
                bag_first_move.append((b,m))
        # Case 1: Victory found during expansion
        if solution_path:
            board_after_first_step = solution_path[1]
            selected_move_string,moves_to_print_string  = first_step_options_string(board_after_first_step,bag_first_move)
            show_solution(solution_path, 4,moves_to_print_string,selected_move_string , detail_output)
            return

        # Case 2: Stuck (no valid moves from any board)
        if not all_candidates:
            break

        # Keep only the best k candidates for the next round
        current_states = select_best_k(all_candidates, k)

    # 3. If loop finishes without success
    show_solution("No path found", 4, None, None, False)


def generate_all_candidates(current_states, goals):
    """
    Expands all states in the current beam.
    Returns: (solution_path, all_candidates_list)
    If a solution is found, solution_path is returned immediately.
    """
    all_candidates = []

    # Iterate over each board in our current beam
    for h_score, board, path,move in current_states:
        moves = get_legal_moves(board)

        for m in moves:
            # Create the new board
            next_board = apply_move(board, m)
            h_new = heuristic(next_board, goals)

            # Create the new path (Copy old path + add new step)
            new_path = list(path)
            new_path.append(next_board)

            # CHECK VICTORY: If we found a solution, return immediately
            if h_new == 0:
                return new_path, all_candidates

            # Add this candidate to the big pool
            # Tuple structure: (Score, Board, Path)
            all_candidates.append((h_new, next_board, new_path,m))

    return None, all_candidates


def select_best_k(all_candidates, k):
    """
    Sorts all candidates and returns the top k.
    """
    # Sort all candidates by score (Ascending: 0 is best)
    # Python sorts tuples by the first element automatically.
    all_candidates.sort()

    # Return only the best k
    return all_candidates[:k]






def first_step_options_string(board_after_first_step, bag_first_move):
    """
    Helper function to format the moves of the first step for display.
    Converts internal 0-based indices to user-friendly 1-based strings.
    """
    selected_move = None

    # 1. Identify the specific move that led to the chosen board state
    # We iterate through the history (bag) to match the resulting board.
    for board, m in bag_first_move:
        if board == board_after_first_step:
            selected_move = m

    # 2. Extract all raw move tuples from the bag (candidates)
    moves_to_print = []
    for board, move in bag_first_move:
        moves_to_print.append(move)

    # 3. Format all candidate moves into strings (e.g., "(1,1) to (2,2)")
    moves_to_print_string = []
    for m in moves_to_print:
        # Convert 0-based index to 1-based index for display
        r1, c1, r2, c2 = m[0] + 1, m[1] + 1, m[2] + 1, m[3] + 1
        string_move = f"({r1},{c1}) to ({r2},{c2})"
        moves_to_print_string.append(string_move)

    # 4. Format the actually selected move similarly
    r1, c1, r2, c2 = selected_move[0] + 1, selected_move[1] + 1, selected_move[2] + 1, selected_move[3] + 1
    selected_move_string = f"({r1},{c1}) to ({r2},{c2})"

    return selected_move_string, moves_to_print_string











# --- Main Solver Function ---

def solve_genetic(starting_board, goal_board, detail_output):
    """
    Main Genetic Algorithm Manager.
    Structure:
    1. Initial Pop = 10
    2. High Crossover rate (via merge_paths), Low Mutation rate.
    """
    global ORIGINAL_MAP
    ORIGINAL_MAP = copy.deepcopy(starting_board)

    # --- Configuration ---
    POP_SIZE = 10
    GENS = 200
    ELITISM_COUNT = 2  # Always keep the top 2
    MUTATION_RATE = 0.2  # Low probability for mutation
    # (The rest, 0.8, goes to Crossover)

    goals = get_horses(goal_board)

    # 1. Create initial population of 10
    population = create_initial_population(starting_board, goals, POP_SIZE)
    founding_population=[]
    for gen in range(GENS):
        # Sort population by heuristic (lowest is best)
        population.sort(key=lambda x: x[0])

        # Check for victory in the best individual
        best_h, best_board, best_path = population[0]
        if best_h == 0:
            if gen == 0:
                founding_population = population
            show_solution(best_path, 5,founding_population , None, detail_output)
            return

        # Create Next Generation
        next_generation = []

        # 1. Elitism: Keep the best ones as they are
        next_generation.extend(population[:ELITISM_COUNT])

        # 2. Fill the rest of the population
        while len(next_generation) < POP_SIZE:

            # Dice roll: Mutation or Crossover?
            if random.random() < MUTATION_RATE:
                # MUTATION
                # Select a parent and add a random step
                parent = select_parent_roulette(population)
                child = mutate_individual(parent, goals)
                if len(child[2]) < 200:
                    # Check for immediate win
                    if child[0] == 0:
                        if gen == 0:
                            founding_population = population
                        show_solution(child[2], 5, founding_population, None, detail_output)
                        return
                    next_generation.append(child)

            else:
                #  CROSSOVER
                # Select two parents and try to merge their paths
                p1 = select_parent_roulette(population)
                p2 = select_parent_roulette(population)

                merged_path = merge_paths(p1[2], p2[2])

                if merged_path:
                    # Crossover successful
                    final_board = merged_path[-1]
                    h_new = heuristic(final_board, goals)
                    if len(merged_path) < 200:
                        child = (h_new, final_board, merged_path)

                        if h_new == 0:
                            if gen==0:
                                founding_population = population
                            show_solution(merged_path, 5, founding_population, None, detail_output)
                            return

                        next_generation.append(child)
                else:
                    # If Crossover failed (no common point), fallback to mutation
                    # so we don't waste the turn
                    child = mutate_individual(p1, goals)
                    next_generation.append(child)

        # Update population for the next generation
        population = next_generation
        if gen==0:
            founding_population=population

    # If loop finishes without success
    show_solution("No path found", 5, None, None, False)


# --- Helper Logic Functions ---

def create_initial_population(start_board, goals, pop_size):
    """
    Generates a diverse initial population by performing random walks
    of varying lengths from the start position.
    """
    population = []

    # 1. Always include the clean starting state as an anchor
    h_start = heuristic(start_board, goals)
    population.append((h_start, start_board, [start_board]))

    attempts = 0
    # Try to fill the population up to POP_SIZE.
    # We use an attempts counter to avoid infinite loops if the board is very small/locked.
    while len(population) < pop_size and attempts < pop_size * 5:
        attempts += 1

        curr_board = copy.deepcopy(start_board)
        curr_path = [curr_board]

        # Choose a random walk length between 1 and 2 steps.
        # This ensures candidates end up in different positions on the board,
        # creating the necessary genetic diversity.
        walk_length = random.randint(1, 2)

        valid_walk = True
        for i in range(walk_length):
            moves = get_legal_moves(curr_board)
            if not moves:
                valid_walk = False
                break

            # Perform a random move
            move = random.choice(moves)
            curr_board = apply_move(curr_board, move)
            curr_path.append(curr_board)

        if valid_walk:
            # Calculate heuristic for the final state of the walk
            h = heuristic(curr_board, goals)

            # Add the new individual to the population
            # Structure: (score, board_state, full_path)
            population.append((h, curr_board, curr_path))

    # If we couldn't generate enough individuals , fill the rest with duplicates
    while len(population) < pop_size:
        population.append(copy.deepcopy(population[-1]))

    return population

def select_parent_roulette(population):
    """
    Selects one parent from the population using Roulette Wheel selection.
    Individuals with lower Heuristic (better) have higher weight.
    """
    weights = []
    for p in population:
        # Invert the heuristic score so that lower values get higher priority. A small epsilon is added to prevent division by zero.
        w = 1.0 / (p[0] + 0.1)
        weights.append(w)

    # random.choices returns a list, so we take [0] to get the item
    return random.choices(population, weights=weights, k=1)[0]


def mutate_individual(individual, goals):
    """
    Mutation: Takes a path and adds one random valid step to its end.
    """
    h, board, path = individual
    moves = get_legal_moves(board)

    if not moves:
        return individual  # No moves possible, return as is

    # Pick a random move
    m = random.choice(moves)
    child_b = apply_move(board, m)
    child_p = path + [child_b]
    child_h = heuristic(child_b, goals)

    return child_h, child_b, child_p


def merge_paths(path1, path2):
    """
    Crossover Operator:
    Attempts to merge two paths if they share a common board state.
    """
    path1_map = {}

    # Map the first path for O(1) lookup
    for i, board in enumerate(path1):
        if i > 0:
            b_tuple = board_to_tuple(board)
            path1_map[b_tuple] = i

    # Iterate through second path to find an intersection
    for i, board in enumerate(path2):
        if i > 0:
            b_tuple = board_to_tuple(board)
            if b_tuple in path1_map:
                cut_path1 = path1_map[b_tuple]
                cut_path2 = i
                # Return the merge path: Head of P1 + Tail of P2
                return path1[:cut_path1 + 1] + path2[cut_path2 + 1:]
    return None


def board_to_tuple(board):
    """ Converts a board (list of lists) into a tuple for hashing/map keys. """
    temp_rows = []
    for row in board:
        temp_rows.append(tuple(row))
    return tuple(temp_rows)




































# --- 5. PRINTING ---

def print_board(board, title=""):
    """
    Utility to print the board exactly as shown in the requirements.
    """
    chars = {0: " ", 1: "@", 2: "&", 3: "*", 4: "="}

    # Print the Title (e.g., "Board 1 (starting position):")
    if title:
        print(f"{title}:")

    # Print the column numbers
    print("   1 2 3 4 5 6")

    # Print rows
    for i in range(len(board)):
        row_str = f"{i + 1}: "
        # Create the row string with mapped characters
        vals = [chars.get(x, '?') for x in board[i]]
        row_str += " ".join(vals)  # Single space between chars based on the image visual
        print(row_str)

    # Separator line at the bottom
    print("-----")


def show_solution(path, Algo_num,val1,val2,detail_output):
    """
    Prints the solution path with exact formatting.
    No summary statistics (steps/cost) at the end.
    """

    # 1. Handle "No path found"
    # If the path is a string (error message), just print it.
    if not isinstance(path, list):
        print("No path found")
    else:

        # 2. Loop through the path
        total_boards = len(path)

        for i, b in enumerate(path):
            board_num = i + 1

            # Determine the exact title format based on the index
            if i == 0:
                title = f"Board {board_num} (starting position)"
            elif i == total_boards - 1:
                title = f"Board {board_num} (goal position)"
            else:
                title = f"Board {board_num}"

            # Print the board
            print_board(b, title)
            if i==1 and detail_output:
                if Algo_num == 1:
                    print_heuristic(val1)
                if Algo_num == 3:
                    print_Probability(val1,val2)
                if Algo_num == 4:
                    print_bag(val1,val2)
                if Algo_num == 5:
                    print_pop(val1)

def print_heuristic(val1):
    if isinstance(val1,float):
        print("Heuristic: " + str(val1))

def print_Probability(val1,val2):
    if val2:
        for res in val2:
            move, h, prob = res

            r1, c1, r2, c2 = move[0] + 1, move[1] + 1, move[2] + 1, move[3] + 1
            string_move = f"({r1},{c1}) to ({r2},{c2})"


            print(f"Check: {string_move} | Heuristic: {h} | Probability: {prob}")

    if val1:
        r1, c1, r2, c2 = val1[0] + 1, val1[1] + 1, val1[2] + 1, val1[3] + 1
        selected_string = f"({r1},{c1}) to ({r2},{c2})"
        print(f"--> Final Selected Move: {selected_string}")

    print("-------------------------------------------\n")


def print_bag(val1,val2):
    if val1 is not None and val2 is not None and len(val1) > 0:
        print("STEP ONE BAG OF MOVES: " + str(val1) + " and the chosen move: " + str(val2))

def print_pop(val1):
    if isinstance(val1,list) and len(val1) > 0:
        for i, individual in enumerate(val1):
            score = individual[0]
            current_board = individual[1]

            title_text = f"Candidate #{i + 1} | Score (H): {score}"

            print_board(current_board, title_text)

# ==========================================
#  PART A: Boards for A* (A-Star) Checks
#  Goal: Test shortest path finding and cost handling (e.g., Swamps)
# ==========================================

# 1. Swamp Test - Can the algorithm handle "expensive" terrain?
# The horse must decide: go through the Swamp (cost 2) or bypass it?
OVER_SWAMP_START = [
    [3, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
OVER_SWAMP_GOAL = [
    [0, 0, 0, 0, 3, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

# 2. General Board - Complex board to test overall pathfinding logic
STARTING_BOARD = [
    [3, 0, 3, 0, 2, 0],
    [0, 0, 0, 3, 1, 4],
    [1, 0, 2, 0, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [2, 0, 0, 4, 0, 4],
    [0, 1, 0, 2, 0, 0]
]
GOAL_BOARD = [
    [0, 0, 0, 0, 2, 3],
    [0, 0, 0, 0, 1, 4],
    [1, 0, 2, 0, 0, 0],
    [0, 0, 1, 0, 1, 3],
    [2, 0, 3, 4, 0, 4],
    [0, 1, 0, 2, 0, 0]
]

# 3. Heuristic Trap - A* should solve this, but Hill Climbing might fail.
# It looks like there's a short path, but it's blocked. A detour is needed.
STARTING_BOARD_TRAP = [
    [3, 0, 0, 1, 0, 0],  # H1
    [0, 1, 1, 1, 1, 0],
    [0, 1, 3, 0, 0, 0],  # H2
    [1, 1, 1, 1, 1, 1],  # Full horizontal wall barrier
    [0, 1, 0, 1, 4, 0],
    [0, 0, 0, 0, 0, 4]
]
GOAL_BOARD_TRAP = [
    [0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 3, 1, 4, 0],
    [0, 0, 3, 0, 0, 4]
]


# ==========================================
#  PART B: Boards for Hill Climbing Checks
#  Goal: Test speed on simple tasks and identify local minima failures
# ==========================================

# 1. Sanity Check - Direct path, no obstacles.
# Hill Climbing should solve this the fastest.
test_board_easy = [
    [3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
test_board_GOAL = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0]
]

# 2. Impossible/Trapped Check
# The rider is completely enclosed. HC should get stuck at the closest point or fail.
test3_start = [
    [3, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 1, 1, 4, 1, 0], # Rider is trapped
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
test3_goal = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 4, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

# 3. Dynamic Obstacles
# One horse blocks another. HC might struggle if it's too greedy.
test4_start = [
    [3, 0, 0, 0, 3, 4],
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
test4_goal = [
    [0, 0, 0, 0, 3, 4],
    [1, 1, 1, 1, 0, 3],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]


# ==========================================
#  PART C: Boards for Simulated Annealing & Genetic Algorithm
#  Goal: Complex problems, multiple horses, escaping local minima
# ==========================================

# 1. Escape Local Minima
# The only path requires moving "backwards" (away from goal) to bypass a wall.
# SA excels here; HC will fail.
sa_force_prob_start = [
    [0, 3, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0], # Blocking optimal move
    [1, 0, 0, 0, 0, 0], # Blocking optimal move
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
sa_force_prob_goal = [
    [3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

# 2. The Crowd Problem - Classic SA Test
# 3 horses need to pass through a narrow corridor. They will block each other.
# The algorithm must move one horse back to let another pass.
sa_test3_start = [
    [3, 0, 3, 0, 3, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0], # Narrow corridor
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [4, 0, 4, 0, 4, 0]
]
sa_test3_goal = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [3, 0, 3, 0, 3, 0],
    [4, 0, 4, 0, 4, 0]
]

# 3. Crossing Paths - Great for Genetic Algorithm
# Horses need to swap sides symmetrically. GA can find creative parallel solutions.
test5_start = [
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [3, 0, 0, 0, 0, 3], # Horses facing each other
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [4, 0, 0, 0, 0, 4]
]
test5_goal = [
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [4, 3, 0, 0, 3, 4] # Sides swapped
]

# 4. Maze and Traps
test7_start = [
    [1, 1, 4, 1, 1, 1],
    [1, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0],
    [1, 1, 3, 1, 0, 1],
    [1, 1, 1, 1, 0, 1]
]
test7_goal = [
    [1, 1, 4, 1, 1, 1],
    [1, 1, 3, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 1]
]
find_path(sa_test3_start,sa_test3_goal,3,True)
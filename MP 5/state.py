import copy
import math
from itertools import count

# NOTE: using this global index means that if we solve multiple
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()


# TODO VI
# Euclidean distance between two state tuples, of the form (x,y, shape)
def euclidean_distance(a, b):
    a_x, a_y = a[0], a[1]
    b_x, b_y = b[0], b[1]
    d = math.pow(a_x - b_x, 2) + math.pow(a_y - b_y, 2)
    return math.sqrt(d)

from abc import ABC, abstractmethod

class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0., use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass

    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass

    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # __hash__ method allow us to keep track of which
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass

    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass


# State: a length 3 list indicating the current location in the grid and the shape
# Goal: a tuple of locations in the grid that have not yet been reached
#   NOTE: it is more efficient to store this as a binary string...
# maze: a maze object (deals with checking collision with walls...)
class MazeState(AbstractState):
    def __init__(self, state, goal, dist_from_start, maze, use_heuristic=True):
        # NOTE: it is technically more efficient to store both the mst_cache and the maze_neighbors functions globally,
        #       or in the search function, but this is ultimately not very inefficient memory-wise
        self.maze = maze
        self.maze_neighbors = maze.get_neighbors

        super().__init__(state, goal, dist_from_start, use_heuristic)

    # TODO VI
    def get_neighbors(self):
        # if the shape changes, it will have a const cost of 10.
        # otherwise, the move cost will be the euclidean distance between the start and the end positions
        nbr_states = []
        neighboring_locs = self.maze_neighbors(*self.state)

        for neighbor in neighboring_locs:

            distance = self.dist_from_start

            if self.state[2] != neighbor[2]:
                distance += 10
            else:
                distance += euclidean_distance(self.state, neighbor)

            new_state = MazeState (
                    state = neighbor,
                    goal = self.goal,
                    dist_from_start = distance,
                    use_heuristic = self.use_heuristic,
                    maze = self.maze,
                )
            
            nbr_states.append(new_state)
        
        return nbr_states

    # TODO VI
    def is_goal(self):
        loc = (self.state[0], self.state[1])
        return (loc in self.goal)

    # We hash BOTH the state and the remaining goals
    #   This is because (x, y, h, (goal A, goal B)) is different from (x, y, h, (goal A))
    #   In the latter we've already visited goal B, changing the nature of the remaining search
    # NOTE: the order of the goals in self.goal matters, needs to remain consistent
    # TODO VI
    def __hash__(self):
        return hash((self.state, tuple(sorted(self.goal))))

    # TODO VI
    def __eq__(self, other):
        return (self.state == other.state) and sorted(self.goal) == sorted(other.goal)

    # Our heuristic is: distance(self.state, nearest_goal)
    # We use euclidean distance
    # TODO VI
    def compute_heuristic(self):
        min_goal = float('inf')
        for goal in self.goal:
            min_goal = min(min_goal, euclidean_distance(self.state, goal))
        return min_goal

    # This method allows the heap to sort States according to f = g + h value
    # TODO VI
    def __lt__(self, other):
        curr_cost = self.h + self.dist_from_start
        other_cost = other.h + other.dist_from_start
        if curr_cost < other_cost:
            return True
        elif curr_cost > other_cost:
            return False
        else:
            return self.tiebreak_idx < other.tiebreak_idx


    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)

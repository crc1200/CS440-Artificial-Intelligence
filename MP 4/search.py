import heapq

def best_first_search(starting_state):

    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    result = []
    curr = starting_state
    while frontier:
        curr = heapq.heappop(frontier)
        if curr.is_goal():
            print("line 43")
            result = backtrack(visited_states, curr)
            break
        
        neighbors = curr.get_neighbors()

        for neighbor in neighbors:
            cost = neighbor.dist_from_start
            if neighbor not in visited_states or cost < visited_states[neighbor][1]:
                visited_states[neighbor] = (curr, cost)
                heapq.heappush(frontier, neighbor)

    return result

def backtrack(visited_states, goal_state):
    path = []

    if goal_state not in visited_states:
        return []

    start = goal_state
    while start:
        path.append(start)
        start = visited_states[start][0]

    return path[::-1]
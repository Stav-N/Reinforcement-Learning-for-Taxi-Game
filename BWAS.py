import numpy as np
from heuristics import *
# from utilities import *

from queue import PriorityQueue


def BWAS(start, W, B, heuristic_function, T):
    
    
    state_f = heuristic_function([start])[0]

    goal_reached = False

    open = MyPriorityQueue()
    closed = {}
    UB = np.inf
    LB =  -np.inf
    expansions = 0
    n_start = (state_f, 0, start, None) # (f, g, TopSpinState, parent)
    open.put(n_start)

    while (open.queue._qsize() > 0) & (expansions <= T): 
        generated = []
        batch_expansion = 0
        while (open.queue._qsize() > 0) & (expansions <= T) & (batch_expansion < B): 
            state_f, state_g, current_state, state_p = open.get()
            closed[current_state] = (state_g, current_state, state_p)
            expansions += 1
            batch_expansion += 1
            if not len(generated):
                LB = max(LB, state_f)
            if current_state.is_goal():
                if UB > state_g:
                    UB = state_g
                    goal_state = current_state 
                    goal_reached = True
                continue
            neighbors = current_state.get_neighbors(heuristic_function)
            for state_tag, h_value in neighbors:
                g_tag = state_g + 1
                if (state_tag not in closed) or (g_tag < closed[state_tag][0]):
                    # befor asighning, check if it create a loop (if it does, remove )
                    closed[state_tag] = (g_tag, state_tag, current_state)
                    generated.append((state_tag, g_tag, current_state, h_value))
        if LB >= UB:
            return reconstruct_the_path(goal_state, closed), expansions
        for state_tag, g_tag, parent_state, h_value in generated:
            f_value = g_tag + W * h_value
            open.put((f_value, g_tag, state_tag, parent_state))
    
    if goal_reached:
        return reconstruct_the_path(goal_state, closed), expansions 
    else: 
        return None, expansions

class MyPriorityQueue():
    def __init__(self):
        self.queue = PriorityQueue()
        self.elements = []

    def get(self):
        """
        pops element from the top of the queue
        """
        item = self.queue.get()
        self.elements.remove(item)
        return item 
    
    def put(self, item):
        """
        put element in the queue
        """
        self.queue.put(item)
        self.elements.append(item)

def reconstruct_the_path(current_state, closed):
    path_to_goal = [current_state]
    _, _, parent_state = closed[current_state]
    while parent_state:
        if parent_state in path_to_goal:
            path_to_goal = path_to_goal[:path_to_goal.index(parent_state)]
            print(f'found loop {parent_state.get_state_as_list()} located at {len(path_to_goal)}')
            return None
        path_to_goal.append(parent_state)
        _, _, parent_state = closed[parent_state]
    path_to_goal.reverse()
    return path_to_goal




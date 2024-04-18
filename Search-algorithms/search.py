# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    visited = set()
    node = problem.getStartState()
    stack = util.Stack()
    # keeping track of actions needed to reach the current node.(path)
    stack.push((node, []))
    while not stack.isEmpty():
        current, exploredActions = stack.pop()
        # I add the unvisited node to the set
        if current not in visited:
            visited.add(current)
            # Next I check if it's the goal state if yes I return the actions needed to reach the goal.
            if problem.isGoalState(current):
                return exploredActions
            # print("Current", current)
            neighborsList = problem.getSuccessors(current)
            # print("Successors", neighborsList)
            for neighbor, action, cost in neighborsList:
                # Next I add the neighbors to the stack to explore them next.
                if neighbor not in visited:
                    stack.push((neighbor, exploredActions + [action]))
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    node = problem.getStartState()
    queue = util.Queue()
    # keeping track of actions needed to reach the current node.(path)
    queue.push((node, []))

    while not queue.isEmpty():
        current, exploredActions = queue.pop()

        # I add the unvisited node to the visited set
        if current not in visited:
            visited.add(current)
            # Next I check if it's the goal state if yes I return the actions needed to reach the goal.
            if problem.isGoalState(current):
                return exploredActions

            # print("Current", current)
            neighborsList = problem.getSuccessors(current)
            # print("Successors", neighborsList)
            for neighbor, direction, cost in neighborsList:
                if neighbor not in visited:
                    # Next I add the neighbors to the queue to explore them next.
                    queue.push((neighbor, exploredActions + [direction]))
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    node = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    # keeping track of actions needed to reach the current node(path), and the cost of the path.
    priorityQueue.push((node, [], 0), 0)

    while not priorityQueue.isEmpty():
        current, exploredActions, cost = priorityQueue.pop()
        # I add the unvisited node to the visited set
        if current not in visited:
            visited.add(current)
            # Next I check if it's the goal state if yes I return the actions needed to reach the goal.
            if problem.isGoalState(current):
                return exploredActions
            # print("Current", current)
            neighborsList = problem.getSuccessors(current)
            # print("Successors", neighborsList)
            for neighbor, direction, neighborCost in neighborsList:
                if neighbor not in visited:
                    """ Next I add the neighbors,the explored actions, the path cost, 
                        to the priorityQueue to explore them next."""
                    priorityQueue.push((neighbor, exploredActions + [direction], cost + neighborCost),
                                       cost + neighborCost)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    node = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    fn = problem.getCostOfActions([]) + heuristic(node, problem)
    # keeping track of actions needed to reach the current node(path), and the fn of the path.
    # f(n) = g(n) + h(n)
    priorityQueue.push((node, [], fn), fn)

    while not priorityQueue.isEmpty():
        current, exploredActions, pathCost = priorityQueue.pop()
        """ I add the unvisited node to the visited set """
        if current not in visited:
            visited.add(current)
            # Next I check if it's the goal state if yes I return the actions needed to reach the goal.
            if problem.isGoalState(current):
                return exploredActions

            # print("Current", current)
            neighborsList = problem.getSuccessors(current)
            # print("Successors", neighborsList)
            for neighbor, direction, neighborCost in neighborsList:
                fn = problem.getCostOfActions(exploredActions + [direction]) + heuristic(neighbor, problem)
                if neighbor not in visited:
                    # Next I add the neighbors,the explored actions, fn to the priorityQueue to explore them next.
                    priorityQueue.push((neighbor, exploredActions + [direction], fn), fn)
                elif neighbor in visited and fn < pathCost:
                    # To break the ties I compare the fn value
                    priorityQueue.push((neighbor, exploredActions + [direction], fn), fn)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

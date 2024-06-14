# AI
This projects is an implementation of Pacman game using different AI algorithms. The [algorithms](#algorithms) implemented in search.py of each folder are used to find a path through the maze. 

## Tech Stack

 [![Python](https://skillicons.dev/icons?i=py)](https://www.python.org/)

## Development Environment
[![My Skills](https://skillicons.dev/icons?i=pycharm)](https://www.jetbrains.com/pycharm/)
  
## Algorithms
BFS<br>
Depth First Search<br>
Uniform Cost Search<br>
A* Search<br>
Alpha-beta pruning<br>
Minimax<br>
Expectimax<br>
Markov decision process - Value iteration, Policy Iteration, Prioritized Sweeping Value Iteration<br>
Reinforcement Learning - Q-learning, Approximate Q-Learning<br>

## How to run these algorithms.
- First clone this repository.
- Open a terminal in your project folder. Note: Each algorithm needs to be run in its respective sub folder.
- Run the following commands to see the results.

## 1. Search Algorithm

### Breadth First Search to Find a Path to a Fixed Object
This graph search algorithm that avoids expanding any already visited states. 

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```

### Depth First Search to Find a Path to a Fixed Object
This graph search algorithm that avoids expanding any already visited states. 

```bash
python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
```

### Uniform-Cost Search to Find a Path to a Fixed Object
By changing the cost function, we can encourage Pacman to find different paths. For example, we can charge more for dangerous steps in ghost-ridden areas or less for steps in food-rich areas, and a rational Pacman agent should adjust its behavior in response.

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
```
```bash
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
```
### A* Search
A* takes a heuristic function as an argument.

```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### A* Search to Find All the Corners
`CornersProblem` search problem in `searchAgents.py` finds the shortest path through the maze that touches all four corners 
 
```bash
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```

### Corners Problem: Heuristic

```bash
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```

### Eating All The Dots :

`foodHeuristic` helps Pacman eat all the food dots in as few steps as possible.

```bash
python pacman.py -l trickySearch -p AStarFoodSearchAgent
```

## 2. Multiagent Pacman

### Reflex Agent 

```bash
python pacman.py --frameTime 0 -p ReflexAgent -k 2
```

### Minimax Agent

```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```

### Alpha-Beta Pruning

```bash
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

### Expectimax 
```bash
python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
```

## 3. MDP and Reinforcement Learning

### Value Iteration
```bash
python gridworld.py -a value -i 100 -k 10
```

## Bridge Crossing Analysis
```bash
python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2
```

## Asynchronous Value Iteration
```bash
python gridworld.py -a asynchvalue -i 1000 -k 10
```
## Prioritized Sweeping Value Iteration
```bash
python gridworld.py -a priosweepvalue -i 1000
```

## Q-Learning
```bash
python gridworld.py -a q -k 5 -m
```

## Approximate Q-Learning
```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic 
```


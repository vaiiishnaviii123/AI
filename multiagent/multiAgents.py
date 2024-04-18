# multiAgents.py
# --------------
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
import random
import util
from math import inf
from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def pacmanRelativeDistance(self, pacmanPosition, nextPosition):
        "The Manhattan distance heuristic for a PositionSearchProblem"
        xy1 = pacmanPosition
        xy2 = nextPosition
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0

        # Checking distance between Pacman and food pellets adding a higher value to score if
        # pacman ate a food pellet
        foodPellets = currentGameState.getFood().asList()
        minDis = float(inf)
        for food in foodPellets:
            minDis = min(minDis, self.pacmanRelativeDistance(newPos, food))

        if minDis == 0:
            score += 100
        else:
            score += 1 / minDis

        # checking distance between Pacman and  ghosts and decreasing score if ghost is near
        distToGhost = None
        nearestGhost = float(inf)
        scaredTime = None
        for ghostPos in newGhostStates:
            distToGhost = self.pacmanRelativeDistance(newPos, ghostPos.getPosition())
            if distToGhost < nearestGhost:
                nearestGhost = distToGhost
                scaredTime = ghostPos.scaredTimer

        if distToGhost <= 1 and scaredTime > 0:
            score += 100
        # Pacman to stay away from ghosts.
        elif distToGhost <= 1 and scaredTime <= 0:
            score -= 100

        # Distance from pacman to power pellet
        nearestCapsuleDist = float(inf)
        powerPellet = currentGameState.getCapsules()
        for capsule in powerPellet:
            nearestCapsuleDist = min(nearestCapsuleDist, self.pacmanRelativeDistance(newPos, capsule))
        # adding a higher value to score if pacman ate a capsule
        if nearestCapsuleDist == 0:
            score += 100
        else:
            score += (1 / nearestCapsuleDist) * 5
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # minimax code - starting with -infinite and then updating the score with max score
        minimaxScore = float('-inf')  # score of all chance nodes
        minimaxAction = None
        legalActions = gameState.getLegalActions(0)  # exploring all actions and storing the best action
        for action in legalActions:
            successors = gameState.generateSuccessor(0, action)
            score = self.minimax(0, 0, successors)
            if score > minimaxScore:
                minimaxScore = score
                minimaxAction = action
        return minimaxAction

    def minimax(self, indexOfAgent, depthOfTree, nextGameState):
        currIndex = indexOfAgent + 1
        # updating agent index and checking if next node is pacman
        if indexOfAgent + 1 == nextGameState.getNumAgents():
            currIndex = 0
        # calculating max score if agent is pacman
        if currIndex == 0:
            depthOfTree += 1
            bestScore = float(-inf)
            maxDepth = depthOfTree == self.depth
            gameOver = nextGameState.isLose() or nextGameState.isWin()
            # checking if end of games if yes return score
            if maxDepth or gameOver:
                return self.evaluationFunction(nextGameState)
            legalActions = nextGameState.getLegalActions(currIndex)
            for action in legalActions:
                nextSuccessors = nextGameState.generateSuccessor(currIndex, action)
                bestScore = max(bestScore, self.minimax(currIndex, depthOfTree, nextSuccessors))
            return bestScore
        # calculating min score if agent is ghost
        maxDepth = depthOfTree == self.depth
        gameOver = nextGameState.isLose() or nextGameState.isWin()
        # checking if end of games if yes return score
        if maxDepth or gameOver:
            return self.evaluationFunction(nextGameState)
        bestScore = float(inf)
        legalActions = nextGameState.getLegalActions(currIndex)
        for action in legalActions:
            nextSuccessors = nextGameState.generateSuccessor(currIndex, action)
            bestScore = min(bestScore, self.minimax(currIndex, depthOfTree, nextSuccessors))
        return bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        maxAction = None
        legalActions = gameState.getLegalActions(0)  # exploring actions and returning the max
        for action in legalActions:
            successors = gameState.generateSuccessor(0, action)
            v = self.minimax(0, 0, successors, alpha, float('inf'))
            if v > alpha:
                alpha = v
                maxAction = action
        return maxAction

    def minimax(self, indexOfAgent, depthOfTree, nextGameState, alpha, beta):
        currIndex = indexOfAgent + 1
        # updating agent index and checking if next node is pacman
        if indexOfAgent + 1 == nextGameState.getNumAgents():
            currIndex = 0
        # calculating max value using minimax if agent is pacman and performing
        # the beta testing as its max node
        if currIndex == 0:
            depthOfTree += 1
            v = float(-inf)
            maxDepth = depthOfTree == self.depth
            gameOver = nextGameState.isLose() or nextGameState.isWin()
            # checking if end of games if yes return score
            if maxDepth or gameOver:
                return self.evaluationFunction(nextGameState)
            legalActions = nextGameState.getLegalActions(currIndex)
            for action in legalActions:
                nextSuccessors = nextGameState.generateSuccessor(currIndex, action)
                v = max(v, self.minimax(currIndex, depthOfTree, nextSuccessors, alpha, beta))
                # beta testing return v if test passed else update alpha value
                if v > beta:
                    return v
                alpha = max(v, alpha)
            return v
        # calculating min value using minimax if agent is ghost and performing
        # the alpha testing as its min node
        maxDepth = depthOfTree == self.depth
        gameOver = nextGameState.isLose() or nextGameState.isWin()
        # checking if end of games if yes return score
        if maxDepth or gameOver:
            return self.evaluationFunction(nextGameState)
        v = float(inf)
        legalActions = nextGameState.getLegalActions(currIndex)
        for action in legalActions:
            nextSuccessors = nextGameState.generateSuccessor(currIndex, action)
            v = min(v, self.minimax(currIndex, depthOfTree, nextSuccessors, alpha, beta))
            # alpha testing return v if test passed else update beta value
            if v < alpha:
                return v
            beta = min(v, beta)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        expectimaxScore = float('-inf')  # score of all chance nodes
        expectimaxAction = None
        legalActions = gameState.getLegalActions(0)  # like right left middle etc
        for action in legalActions:
            successors = gameState.generateSuccessor(0, action)
            score = self.expectimax(0, 0, successors)
            if score > expectimaxScore:
                expectimaxScore = score
                expectimaxAction = action
        return expectimaxAction

    def expectimax(self, indexOfAgent, depthOfTree, nextGameState):
        # updating agent index and checking if next node is pacman
        currIndex = indexOfAgent + 1
        if indexOfAgent + 1 == nextGameState.getNumAgents():
            currIndex = 0
        # calculating max value using expectimax if agent is pacman
        if currIndex == 0:
            depthOfTree += 1
            bestScore = float('-inf')
            maxDepth = depthOfTree == self.depth
            gameOver = nextGameState.isLose() or nextGameState.isWin()
            # checking if end of games if yes return score
            if maxDepth or gameOver:
                return self.evaluationFunction(nextGameState)
            legalActions = nextGameState.getLegalActions(currIndex)
            for action in legalActions:
                nextSuccessors = nextGameState.generateSuccessor(currIndex, action)
                bestScore = max(bestScore, self.expectimax(currIndex, depthOfTree, nextSuccessors))
            return bestScore
        # calculating average value using expectimax if agent is ghost. (chance node value)
        maxDepth = depthOfTree == self.depth
        gameOver = nextGameState.isLose() or nextGameState.isWin()
        # checking if end of games if yes return score
        if maxDepth or gameOver:
            return self.evaluationFunction(nextGameState)
        score = 0
        cnt = 0
        legalActions = nextGameState.getLegalActions(currIndex)
        for action in legalActions:
            nextSuccessors = nextGameState.generateSuccessor(currIndex, action)
            currScore = self.expectimax(currIndex, depthOfTree, nextSuccessors)
            cnt = cnt + 1
            score = score + currScore
        chanceNodeValue = score / (1.0 * cnt)
        return chanceNodeValue


def pacmanRelativeDistance(pacmanPosition, nextPosition):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = pacmanPosition
    xy2 = nextPosition
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I am checking the ghost position first as mentioned in the suggestions if ghost is
    near and scared time left is less than or equal to 0, I add positive infinite to the nearest
    food distance and reduce the score so that pacman runs away from the ghost.
    Next if the ghost is far and scared time is > 0 then I add a positive value to score encouraging
    pacman to explore the nearest food pellets.
    If there are food pellets and capsules yet to be explored I
    add small value to the score so that pacman explores more food.
    """
    "*** YOUR CODE HERE ***"
    currentScore = currentGameState.getScore()
    currentPacmanPosition = currentGameState.getPacmanPosition()
    score = currentScore
    foodPellets = currentGameState.getFood().asList()
    foodPelletCount = currentGameState.getNumFood()
    totalCapsules = len(currentGameState.getCapsules())

    """Checking distance between Pacman and nearest food pellets."""
    minDisToFood = float(inf)
    for food in foodPellets:
        minDisToFood = min(minDisToFood, pacmanRelativeDistance(currentPacmanPosition, food))

    """Checking the nearest ghost distance from current packman position, if ghost too near 
    updating food position to positive infinite"""
    ghostStates = currentGameState.getGhostStates()
    nearestGhost = float(inf)
    scaredTime = None
    for ghostPos in ghostStates:
        distToGhost = pacmanRelativeDistance(currentPacmanPosition, ghostPos.getPosition())
        if distToGhost < nearestGhost:
            nearestGhost = distToGhost
            scaredTime = ghostPos.scaredTimer

    if nearestGhost <= 1 and scaredTime > 0:
        score += 100
    # Pacman to stay away from ghosts and run for life.
    elif nearestGhost <= 1 and scaredTime <= 0:
        score -= 100
        minDisToFood = float('inf')

    """Checking distance to closet food and updating score as below """
    if minDisToFood == 0:
        score += 100
    else:
        score += (1 / minDisToFood) * 1

    """remaining capsules check"""
    if totalCapsules == 0:
        score += 10
    else:
        score += (1 / totalCapsules)

    """ remaining foodCount check"""
    if foodPelletCount == 0:
        score += 10
    else:
        score += (1/foodPelletCount)

    return score


# Abbreviation
better = betterEvaluationFunction

# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        """For every iteration we need to calculate Q Value for each of the states in MDP"""
        for itr in range(self.iterations):
            states = self.mdp.getStates()
            vValues = util.Counter()  # initial v values/ v values from last iteration

            for s in states:
                policyAtS = self.getPolicy(s)
                if policyAtS:
                    vValues[s] = self.getQValue(s, policyAtS)
            self.values = vValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        """Getting the possible transitions from given state and action"""
        stateTransitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0.0
        for transition in stateTransitions:
            sPrime = transition[0]
            t = transition[1]
            r = self.mdp.getReward(state, action, sPrime)
            vPrime = self.getValue(sPrime)
            qValue += t * (r + (self.discount * vPrime))
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        """when state has no available actions its probably a terminal state."""
        if self.mdp.isTerminal(state):
            return None

        """Getting all actions and its probabilities for the given state"""
        availableActions = self.mdp.getPossibleActions(state)
        maxQValue = float('-inf')
        maxPolicy = None
        """Computing the Q values and storing the action for the max qValue."""
        for action in availableActions:
            qValue = self.getQValue(state, action)
            if maxQValue < qValue:
                maxQValue = qValue
                maxPolicy = action
        return maxPolicy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        """For every iteration we need to calculate Q Value only for one state in MDP"""
        states = self.mdp.getStates()
        index = 0

        for itr in range(self.iterations):
            if index == len(states):
                index = 0
            state = states[index]
            policyAtS = self.getPolicy(state)
            if policyAtS:
                self.values[state] = self.getQValue(state, policyAtS)
            index += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        priorityQueue = util.PriorityQueue()
        predecessors = {}
        for state in states:
            availableActions = self.mdp.getPossibleActions(state)
            for action in availableActions:
                stateTransitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition in stateTransitions:
                    sPrime, t = transition
                    if t > 0:
                        if sPrime not in predecessors:
                            predecessors[sPrime] = set()
                        predecessors[sPrime].add(state)

        for state in states:
            if not self.mdp.isTerminal(state):
                diff = abs(self.getValue(state) - self.getMaxQValue(state))
                priorityQueue.push(state, diff*(-1.0))

        for i in range(0, self.iterations):
            if priorityQueue.isEmpty():
                return
            state = priorityQueue.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getMaxQValue(state)

            for p in predecessors[state]:
                diff = abs(self.getValue(p) - self.getMaxQValue(p))
                if diff > self.theta:
                    priorityQueue.update(p, diff*(-1.0))


    def getMaxQValue(self, state):
        maxValue = float('-inf')
        availableActions = self.mdp.getPossibleActions(state)
        for action in availableActions:
            qValue = self.getQValue(state, action)
            maxValue = max(maxValue, qValue)
        return maxValue


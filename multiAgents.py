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


from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print bestScore
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

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
            newFood = successorGameState.getFood().asList()
            newGhostStates = successorGameState.getGhostStates()
            newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

            "*** YOUR CODE HERE ***"
            newFood.extend(currentGameState.getCapsules())

            currentPos = currentGameState.getPacmanPosition()
            successorDistanceFromFood = []
            for a, b in newFood:
                successorDistanceFromFood.append(abs(a - newPos[0]) + abs(b - newPos[1]))
                # print(500 - min(distance))
                # if newPos in currentGameState.getFood().asList():
                #   return 500
            currentDistanceFromFood = []
            for a, b in currentGameState.getFood().asList():
                currentDistanceFromFood.append(abs(a - currentPos[0]) + abs(b - currentPos[1]))
            """
            successorNotScaredGhostPositions = []
            successorScaredGhostPositions = []
            for ghost in successorGameState.getGhostStates():
                a, b = ghost.getPosition()
                if(ghost.scaredTimer == False):
                    successorNotScaredGhostPositions.append(abs(a - newPos[0]) + abs(b - newPos[1]))
                else:
                    successorScaredGhostPositions.append(abs(a - newPos[0]) + abs(b - newPos[1]))
    
            currentNotScaredGhostPositions = []
            currentScaredGhostPositions = []
            for ghost in currentGameState.getGhostStates():
                a, b = ghost.getPosition()
                if (ghost.scaredTimer == False):
                    currentNotScaredGhostPositions.append(abs(a - newPos[0]) + abs(b - newPos[1]))
                else:
                    currentScaredGhostPositions.append(abs(a - newPos[0]) + abs(b - newPos[1]))
            nearestCurrent =0
            """
            newFood.extend(currentGameState.getCapsules())

            #for p in currentGameState.getCapsules():
             #   newFood.append(p)

            successorScaredGhostPositions = []
            for ghost in successorGameState.getGhostStates():
                a, b = ghost.getPosition()
                successorScaredGhostPositions.append(abs(a - newPos[0]) + abs(b - newPos[1]))


            currentScaredGhostPositions = []
            for ghost in currentGameState.getGhostStates():
                a, b = ghost.getPosition()
                currentScaredGhostPositions.append(abs(a - currentPos[0]) + abs(b - currentPos[1]))

            score = 0
            if successorGameState.isWin():
                return 5000
            #if min(currentDistanceFromFood) > min(successorDistanceFromFood):
             #   score += 150
            """
            for i in range(len(currentDistanceFromFood)):
                if currentDistanceFromFood[i] == min(currentDistanceFromFood):
                    nearestCurrent = i
                    break
            score += abs(newPos[0] - )
            """

            if len(newFood) == 0 : return 5000
            #if newPos in currentGameState.getFood().asList() and  : return 5000
            if newPos in currentGameState.getFood().asList():
                score += 500
            #if len(newFood) < len(currentGameState.getFood().asList()):
            #    score += 500

            #if action == Directions.STOP:
             #   score -= 200
            maxScared = max(newScaredTimes)
            #print maxScared
            if maxScared == 0:

                #if len(currentScaredGhostPositions) != 0 and len(successorScaredGhostPositions) != 0:

                        if min(successorScaredGhostPositions) < 2:
                        #if min(successorScaredGhostPositions) > min(currentScaredGhostPositions):
                            score -= 1000
                        else:
                            #print "p"
                            score += 200
                            #print "Current - " + str(min(currentScaredGhostPositions)) + " Successor " + str(min(successorScaredGhostPositions))

            if maxScared > 0:
                if min(successorScaredGhostPositions) < maxScared:
                    if min(successorScaredGhostPositions) < min(currentScaredGhostPositions):
                        score += 1000



            """
            if len(currentNotScaredGhostPositions) != 0 and len(successorNotScaredGhostPositions) != 0:
                if min(successorNotScaredGhostPositions) <= min(currentNotScaredGhostPositions):
                        print "p"
                        score -= 500
                else:
    
                    score += 500
            """
            if newPos in currentGameState.getCapsules():
                score += 1000
            #print min(successorDistanceFromFood)
            score -= min(successorDistanceFromFood)
            #print "l - "+str(len(newFood))+" score- "+str(score)
            return score
            #print currentGameState.getScore()
            #print z
            return successorGameState.getScore()


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"

        numberOfGhosts = gameState.getNumAgents() -1

        def maxValue (gameState, depth):
            depth -= 1
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            maximumValue = -100000

            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                maximumValue = max(maximumValue,minValue(nextState,depth,1))
            return maximumValue

        def minValue (gameState, depth, ghostNumber):
            minimumValue = 100000
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(ghostNumber):
                nextState = gameState.generateSuccessor(ghostNumber, action)
                if ghostNumber == numberOfGhosts:
                    minimumValue = min(minimumValue, maxValue(nextState, depth))
                else:
                    minimumValue = min(minimumValue, minValue(nextState, depth, ghostNumber+1))

            return minimumValue

        maximumValue = -100000
        answer = ''
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            v = minValue(nextState, self.depth,1)
            if v > maximumValue:
                answer = action
                maximumValue = v
        return answer
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numberOfGhosts = gameState.getNumAgents() - 1

        def maxValue(gameState, depth, alpha, beta):
            depth -= 1
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            maximumValue = -100000

            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                maximumValue = max(maximumValue, minValue(nextState, depth, 1 , alpha, beta))
                if maximumValue > beta: return maximumValue
                alpha = max (maximumValue , alpha)
            return maximumValue

        def minValue(gameState, depth, ghostNumber, alpha, beta):
            minimumValue = 100000
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(ghostNumber):
                nextState = gameState.generateSuccessor(ghostNumber, action)
                if ghostNumber == numberOfGhosts:
                    minimumValue = min(minimumValue, maxValue(nextState, depth, alpha, beta))
                else:
                    minimumValue = min(minimumValue, minValue(nextState, depth, ghostNumber + 1, alpha, beta))
                if (minimumValue < alpha): return minimumValue
                beta = min(minimumValue, beta)
            return minimumValue

        maximumValue = -100000
        answer = ''
        alpha = -100000
        beta = 100000
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            v = minValue(nextState, self.depth, 1, alpha, beta)
            if v > maximumValue:
                answer = action
                maximumValue = v
            if maximumValue > beta: return answer
            alpha = max(alpha, maximumValue)
        return answer

        util.raiseNotDefined()

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
        numberOfGhosts = gameState.getNumAgents() - 1

        def maxValue(gameState, depth):
            depth -= 1
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            maximumValue = -100000

            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                maximumValue = max(maximumValue, minValue(nextState, depth, 1))
            return maximumValue

        def minValue(gameState, depth, ghostNumber):
            minimumValue = 100000
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            expectedValue =0
            for action in gameState.getLegalActions(ghostNumber):
                nextState = gameState.generateSuccessor(ghostNumber, action)
                if ghostNumber == numberOfGhosts:
                    minimumValue =  maxValue(nextState, depth)
                else:
                    minimumValue = minValue(nextState, depth, ghostNumber + 1)
                expectedValue += minimumValue
            return float(expectedValue)/float(len(gameState.getLegalActions(ghostNumber)))

        maximumValue = -100000
        answer = ''
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            v = minValue(nextState, self.depth, 1)
            if v > maximumValue:
                answer = action
                maximumValue = v
        return answer
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """
    def breadthFirstSearch((a,b) , (x,y)):
        
        "*** YOUR CODE HERE ***"
        visited = []  # to keep list of visited nodes
        notVisited = util.Queue()
        notVisited.push([(a,b),0])
        def getSuccessors(a,b):
            ans=[]

            if currentGameState.hasWall(a-1,b) == False:
                ans.append((a-1,b))
            if  currentGameState.hasWall(a+1,b) == False:
                ans.append((a+1,b))
            if  currentGameState.hasWall(a,b+1) == False:
                ans.append((a,b+1))
            if  currentGameState.hasWall(a,b-1) == False:
                ans.append((a,b-1))
            return ans

        while not notVisited.isEmpty():

            sPath = notVisited.pop()
            # print "visiting ", sPath
            # print "Spath type", type(sPath)
            s = sPath[0]
            # print "visiting ", s

            if s == (x,y):
                # print "In goal Check for ",s

                return sPath[1]

            if s not in visited:
                visited.append(s)

                for child in getSuccessors(s[0],s[1]):
                    if child not in visited:
                        notVisited.push([child,sPath[1]+1])


    currentPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodList.extend(currentGameState.getCapsules())
    currentDistanceFromFood = []
    scaredTimerGhost =[]
    if len(foodList) == 0: return 1000
    for a, b in foodList:
        #currentDistanceFromFood.append(abs(a - currentPosition[0]) + abs(b - currentPosition[1]))
        currentDistanceFromFood.append(breadthFirstSearch(currentPosition,(a,b)))
    #print currentDistanceFromFood
    minFood = min(currentDistanceFromFood)
    currentDistanceFromGhost = []
    for ghost in currentGameState.getGhostStates():
        a, b = ghost.getPosition()
        currentDistanceFromGhost.append(abs(a - currentPosition[0]) + abs(b - currentPosition[1]))
        #currentDistanceFromGhost.append(breadthFirstSearch(currentPosition,(a,b)))
    minGhost = min(currentDistanceFromGhost)
    for ghostState in currentGameState.getGhostStates():
        scaredTimerGhost.append(ghostState.scaredTimer)
    #score = 1000
    #score = min(currentDistanceFromFood)
    score = currentGameState.getScore()
    score -= len(foodList)
    score -= minFood
    scaredGhost = min(scaredTimerGhost)
    if scaredGhost > 0:
        if minGhost < scaredGhost:
            score += 100
    else:
        if minGhost < 2:
            score += minGhost




    #print score
    return score
    """

    def breadthFirstSearch((a, b), (x, y)):
        """Search the shallowest nodes in the search tree first."""
        "*** YOUR CODE HERE ***"
        visited = []  # to keep list of visited nodes
        notVisited = util.Queue()
        notVisited.push([(a, b), 0])

        def getSuccessors(a, b):
            ans = []

            if currentGameState.hasWall(a - 1, b) == False:
                ans.append((a - 1, b))
            if currentGameState.hasWall(a + 1, b) == False:
                ans.append((a + 1, b))
            if currentGameState.hasWall(a, b + 1) == False:
                ans.append((a, b + 1))
            if currentGameState.hasWall(a, b - 1) == False:
                ans.append((a, b - 1))
            return ans

        while not notVisited.isEmpty():

            sPath = notVisited.pop()
            # print "visiting ", sPath
            # print "Spath type", type(sPath)
            s = sPath[0]
            # print "visiting ", s

            if s == (x, y):
                # print "In goal Check for ",s

                return sPath[1]

            if s not in visited:
                visited.append(s)

                for child in getSuccessors(s[0], s[1]):
                    if child not in visited:
                        notVisited.push([child, sPath[1] + 1])

    currentPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodList.extend(currentGameState.getCapsules())
    currentDistanceFromFood = []
    scaredTimerGhost = []
    if len(foodList) == 0: return 1000
    for a, b in foodList:
         currentDistanceFromFood.append(abs(a - currentPosition[0]) + abs(b - currentPosition[1]))
        #currentDistanceFromFood.append(breadthFirstSearch(currentPosition, (a, b)))
    # print currentDistanceFromFood
    minFood = min(currentDistanceFromFood)
    currentDistanceFromGhost = []
    for ghost in currentGameState.getGhostStates():
        a, b = ghost.getPosition()
        currentDistanceFromGhost.append(abs(a - currentPosition[0]) + abs(b - currentPosition[1]))
        # currentDistanceFromGhost.append(breadthFirstSearch(currentPosition,(a,b)))
    minGhost = min(currentDistanceFromGhost)
    for ghostState in currentGameState.getGhostStates():
        scaredTimerGhost.append(ghostState.scaredTimer)
    # score = 1000
    # score = min(currentDistanceFromFood)
    score = currentGameState.getScore()
    score -= len(foodList)

    score -= float(sum(currentDistanceFromFood))/ float(len(currentDistanceFromFood))
    scaredGhost = min(scaredTimerGhost)
    if scaredGhost > 0:
        if minGhost < scaredGhost:
            score += 100
    else:
        if minGhost < 2:
            score += minGhost

    # print score
    return score



    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


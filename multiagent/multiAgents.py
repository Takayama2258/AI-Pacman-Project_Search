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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score=0
        foods = newFood.asList()
        if len(foods)!=0:
          food_dist = [manhattanDistance(newPos, food) for food in foods]
          minFood = min(food_dist)
          score-=2*minFood
          score-=1000*len(foods)

        ghost_dist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhost = min(ghost_dist)
        score+=minGhost
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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agent, depth):
          result=[]

          if not state.getLegalActions(agent) or depth == self.depth:
            return (self.evaluationFunction(state),0)

          if agent == self.index: #Pacman
            result=[-1e10,'']
            for a in state.getLegalActions(agent):
              s = state.generateSuccessor(agent,a)
              newResult=minimax(s,agent+1,depth)
              oldResult = result[0]
              if newResult[0] > oldResult:
                result[0]=newResult[0]
                result[1]=a
            return result
            
          else: #Ghost
            result=[1e10,'']
            for a in state.getLegalActions(agent):
              s = state.generateSuccessor(agent,a)
              if agent == (state.getNumAgents()-1): #last ghost
                newResult=minimax(s,self.index,depth+1)
              else:
                newResult=minimax(s,agent+1,depth)
              oldResult = result[0]
              if newResult[0] < oldResult:
                result[0]=newResult[0]
                result[1]=a
            return result
        return minimax(gameState, self.index, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def purn(state, agent, depth, a, b):
          result=[]

          if depth == self.depth or (not state.getLegalActions(agent)):
            return (self.evaluationFunction(state),0)

          if agent == self.index: #Pacman
            result=[-1e10,'']
            for action in state.getLegalActions(agent):
              s = state.generateSuccessor(agent,action)
              if result[0] > a:
                return result
              
              oldResult = result[0]
              newResult=purn(s,agent+1,depth,a,b)
              if newResult[0] > oldResult:
                result[0]=newResult[0]
                result[1]=action
                b = max(b,result[0])
                if b > a:
                  return result
            
          if agent != self.index: #Ghost
            result=[1e10,'']
            for action in state.getLegalActions(agent):
              s = state.generateSuccessor(agent,action)
              oldResult = result[0]
              if result[0] < b:
                return result

              if agent == (state.getNumAgents()-1): #last ghost
                newResult=purn(s,self.index,depth+1,a,b)
              else:
                newResult=purn(s,agent+1,depth,a,b)

              if newResult[0] < oldResult:
                result[0]=newResult[0]
                result[1]=action
                a = min(a,result[0])
                if b > a:
                  return result
                
          return result

        return purn(gameState, self.index, 0, 1e11, -1e11)[1]


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
        def exp(state, agent, depth):
          result=[]

          if not state.getLegalActions(agent) or depth == self.depth:
            return (self.evaluationFunction(state),0)

          if agent == self.index: #Pacman
            result=[-1e10,'']
            for a in state.getLegalActions(agent):
              s = state.generateSuccessor(agent,a)
              newResult=exp(s,agent+1,depth)
              oldResult = result[0]
              if newResult[0] > oldResult:
                result[0]=newResult[0]
                result[1]=a
            return result
            
          else: #Ghost
            result=[0,'']
            for a in state.getLegalActions(agent):
              s = state.generateSuccessor(agent,a)
              if agent == (state.getNumAgents()-1): #last ghost
                newResult=exp(s,self.index,depth+1)
              else:
                newResult=exp(s,agent+1,depth)
              oldResult = result[0]
              p = len(state.getLegalActions(agent))
              result[0]=result[0]+(float(newResult[0])/p)
              result[1]=a
            return result
        return exp(gameState, self.index, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <score the foods, capsules, ghost and scared ghosts seperately. For food, capsules and scared ghost, 
      the smaller the total number is the better. I weight the capsule more than the food as eat one capsule will gain more points.
      Closer food, capsules or scared ghost will weight more. For the ghost Pacman should avoid it especially when it is very close, 
      Pacman should try to never be caught by ghost thus when a ghost is very close the state is assigned to a very low score.>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    position = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    foods = currentGameState.getFood().asList()
    score -= len(foods)*10
    foodDist = [manhattanDistance(position, food) for food in foods]
    if foodDist:
      score -= min(foodDist)
    for food in foodDist:
        score+= 5.0/food

    capDist = [manhattanDistance(position, cap) for cap in capsules]
    score -= len(capsules)*40
    if capDist:
      score -= min(capDist)*10

    scareDist = []
    ghostDist = []
    for ghost in ghosts:
      distance = manhattanDistance(position, ghost.getPosition())
      if ghost.scaredTimer:
        scareDist.append(distance)
      else:
        ghostDist.append(distance)
    score -= len(scareDist)*10
    if scareDist:
      score += 50.0/min(scareDist)
    if ghostDist:
      if min(ghostDist)<2:
        score += min(ghostDist)*50
      for dist in ghostDist:
        if dist >0 and dist<7:
          score -= pow(3.0/dist,2)
    
    return score


# Abbreviation
better = betterEvaluationFunction


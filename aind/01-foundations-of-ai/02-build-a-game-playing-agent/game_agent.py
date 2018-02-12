"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

#### some ideas/strategies ####
# minimizing opp moves
# maximizing own moves
# maximizing by minimizing opp moves
# look two moves ahead and keep own moves at max (including -1 moves through opp movement)
# run away strategy
# avoiding corners
# mirroring
# countermirroring
# include 1/8 + 4 start positions, forecast best starting positions (player 1)
# include start positions (player 2)

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


# returns true if the current location of the player is in a corner
def is_corner_location(game, player_location):
    corner_positions = [(0, 0), (0, game.height - 1), (game.width - 1, 0), (game.width - 1, game.height - 1)]
    return player_location in corner_positions

# penalty_heuristic
def penalty_heuristic(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # penalizes a player for moving to corner positions
    penalize_factor = 2
    if is_corner_location(game, game.get_player_location(player)):
        own_moves -= penalize_factor

    return float(own_moves - opp_moves)


# function according to the one in the Board class (__get_moves())
def get_moves(game, loc):
        """Generate the list of possible moves for an L-shaped motion (like a
        knight in chess).
        """
        if loc == game.NOT_MOVED:
            return game.get_blank_spaces()

        r, c = loc
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        valid_moves = [(r + dr, c + dc) for dr, dc in directions
                       if game.move_is_legal((r + dr, c + dc))]
        random.shuffle(valid_moves)
        return valid_moves

# look_ahead_heuristic
def look_ahead_heuristic(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : objects
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_score = 0
    own_moves = game.get_legal_moves(player)
    for move in own_moves:
        own_score += len(get_moves(game, move))
        
    opp_score = 0
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    for move in opp_moves:
        opp_score += len(get_moves(game, move))

    return float(own_score - opp_score)


# look_ahead_heuristic_alt
def look_ahead_alt_heuristic(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')

    own_legal_moves = game.get_legal_moves(player)
    own_moves = len(own_legal_moves)
    for move in own_legal_moves:
        own_moves += len(get_moves(game, move))

    opp_legal_moves = game.get_legal_moves(game.get_opponent(player))
    opp_moves = len(opp_legal_moves)
    for move in opp_legal_moves:
        opp_moves += len(get_moves(game, move))

    return float(own_moves - opp_moves)


# returns the current distance between player and opponent
def get_player_distance(game, player, opp_player):
    player_location = game.get_player_location(player)
    opp_location = game.get_player_location(opp_player)
    x_distance = math.pow(player_location[0] - opp_location[0], 2)
    y_distance = math.pow(player_location[1] - opp_location[1], 2)
    return math.sqrt(x_distance + y_distance)

# run_away_heuristic
def run_away_heuristic(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')

    opp_player = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opp_player))

    # rewards a player for choosing moves away from the opponent's current location
    distance = get_player_distance(game, player, opp_player)
    return float((own_moves + distance) - opp_moves)


# weighted_moves_heuristic
def weighted_moves_heuristic(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height, game.width
    
    score = 0
    own_moves = game.get_legal_moves(player)
    for move in own_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            score += 6
        else:
            score += 8

    return float(score)


# weighted_moves_alt_heuristic
def weighted_moves_alt_heuristic(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height, game.width
    
    own_score = 0
    own_moves = game.get_legal_moves(player)
    for move in own_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            own_score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            own_score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            own_score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            own_score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            own_score += 6
        else:
            own_score += 8

    opp_score = 0           
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    for move in opp_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            opp_score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            opp_score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            opp_score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            opp_score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            opp_score += 6
        else:
            opp_score += 8

    return float(own_score - opp_score)


def custom_score(game, player):
    # TODO: finish this function!
    # return penalty_heuristic(game, player)
    return look_ahead_heuristic(game, player)
    # return run_away_heuristic(game, player)
    # return weighted_moves_heuristic(game, player)
    # return weighted_moves_alt_heuristic(game, player)
    # return look_ahead_alt_heuristic(game, player)

def custom_score_2(game, player):
    # TODO: finish this function!
    # return penalty_heuristic(game, player)
    # return look_ahead_heuristic(game, player)
    # return run_away_heuristic(game, player)
    # return weighted_moves_heuristic(game, player)
    return weighted_moves_alt_heuristic(game, player)
    # return look_ahead_alt_heuristic(game, player)

def custom_score_3(game, player):
    # TODO: finish this function!
    # return penalty_heuristic(game, player)
    # return look_ahead_heuristic(game, player)
    return run_away_heuristic(game, player)
    # return weighted_moves_heuristic(game, player)
    # return weighted_moves_alt_heuristic(game, player)
    # return look_ahead_alt_heuristic(game, player)
                      
class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        # best_move = (-1, -1)
        available_moves = game.get_legal_moves(game._active_player)
        if (len(available_moves) > 0): 
            best_move = available_moves[0]
        else:
            best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float('-inf')
        best_move = None
        
        available_moves = game.get_legal_moves(game._active_player)
        if (len(available_moves) > 0):
            best_move = available_moves[0]
        else:
            best_move = (-1, -1)
        
        # Initialize adversarial opponent move
        game._opponent_moves = game.get_legal_moves(game._inactive_player)
        
        for m in game.get_legal_moves(game._active_player):
            v = self.min_value(game.forecast_move(m), depth-1)
            
            if v > best_score:
                best_score = v
                best_move = m
        return best_move
        
    def min_value(self, game, depth):
        """Helper function for minimization
        """
        # Timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # If last node to evaluate, return the score value
        if depth == 0:
            return self.score(game, self)
        
        # Otherwise keeps going through the tree
        else:
            v = float('inf')
            for m in game.get_legal_moves(game._active_player):
                v = min(v, self.max_value(game.forecast_move(m), depth-1))
            return v
                
    def max_value(self, game, depth):
        """Helper function for maximization
        """
        # Timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # If last node to evaluate, return the score value
        if depth == 0:
            return self.score(game, self)
        
        # Otherwise keeps going through the tree
        else:
            v = float('-inf')
            for m in game.get_legal_moves(game._active_player):
                v = max(v, self.min_value(game.forecast_move(m), depth-1))
            return v


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        # best_move = (-1, -1)
        available_moves = game.get_legal_moves(game._active_player)
        if (len(available_moves) > 0): 
            best_move = available_moves[0]
        else:
            best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while (1>0):
                try:
                    best_move = self.alphabeta(game, depth)
                    depth += 1
                except SearchTimeout:
                    return best_move
            # return self.alphabeta(game, self.search_depth)

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        best_score = float("-inf")
        no_legal_moves = (-1, -1)
   
        if not game.get_legal_moves(game._active_player):
            return no_legal_moves
     
        available_moves = game.get_legal_moves(game._active_player)
        if (len(available_moves) > 0):
            best_move = available_moves[0]

        for move in available_moves:
            value = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if value > best_score:
                best_score = value
                best_move = move
            alpha = max(value, alpha)
        return best_move
    
    
    def min_value(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if not game.get_legal_moves(game._active_player):
           return self.score(game, self)
       
        if depth <= 0:
            return self.score(game, self)
        
        value = float("inf")
        for move in game.get_legal_moves(game._active_player):
            value = min(value, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            # check current value lesser than than the maximum bound alpha in the maximizer function a level up
            if value <= alpha:
                return value
            if value < beta:
                beta = value
        depth -= 1   
        return value

    
    def max_value(self, game, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if not game.get_legal_moves():
            return self.score(game, self)
        
        if depth <= 0:
            return self.score(game, self)
        
        value = float("-inf")
        for move in game.get_legal_moves(game._active_player):
            value = max(value, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))
            # check current value greater than the minimum bound beta in the minimizer function a level up
            if value >= beta:
                return value
            # alpha = max(value, alpha)
            if value > alpha:
                alpha = value
        depth -= 1     
        return value

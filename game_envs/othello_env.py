from player import Player
import numpy as np
from random import choice, choices
from deep_single_agent_env import DeepSingleAgentEnv
import tensorflow as tf

class OthelloEnv(DeepSingleAgentEnv):
    def __init__(self,wht_player_mark=1,blk_player_mark=2):
        self.board_size = 8
        self.whtplayer = Player(wht_player_mark)
        self.blkplayer = Player(blk_player_mark,is_current=True)
        self.board = np.zeros((self.board_size,self.board_size))
        self.board[self.board_size//2-1,self.board_size//2-1] = wht_player_mark
        self.board[self.board_size//2,self.board_size//2] = wht_player_mark
        self.board[self.board_size//2-1,self.board_size//2] = blk_player_mark
        self.board[self.board_size//2,self.board_size//2-1] = blk_player_mark
        self.move_directions = [(-1, -1), (-1, 0), (-1, +1),
                                (0, -1),           (0, +1),
                                (+1, -1), (+1, 0), (+1, +1)]
        self.current_player_mark = self.blkplayer.mark

    def switch_player(self):
        if self.whtplayer.mark == self.current_player_mark:
            self.current_player_mark = self.blkplayer.mark
        else:
            self.current_player_mark = self.whtplayer.mark

    def increase_curr_player_score(self):
        if self.whtplayer.mark == self.current_player_mark:
            self.whtplayer.score += 1
        else:
            self.blkplayer.score += 1

    def decrease_adversary_score(self):
        if self.whtplayer.mark == self.current_player_mark:
            self.blkplayer.score -= 1
        else:
            self.whtplayer.score -= 1

    def apply_move(self,move:list[int]):
        
        if self.is_legal_move(move):
            self.board[move[0]][move[1]] = self.current_player_mark
            self.flip_tiles(move)
            self.switch_player()

    def has_legal_move(self)->bool:
        ''' Method: has_legal_move
            Parameters: self
            Returns: boolean 
                     (True if the player has legal move, False otherwise)
            Does: Checks whether the current player has any legal move 
                  to make.
        '''
        for row in range(self.board_size):
            for col in range(self.board_size):
                move = (row, col)
                if self.is_legal_move(move):
                    return True
        return False

    def is_legal_move(self, move:tuple)->bool:
        ''' Method: is_legal_move
            Parameters: self, move (tuple)
            Returns: boolean (True if move is legal, False otherwise)
            Does: Checks whether the player's move is legal.
                  About input: move is a tuple of coordinates (row, col).
        '''
        if move != () and self.is_valid_coord(move[0], move[1]) and self.board[move[0]][move[1]] == 0:
            for direction in self.move_directions:
                if self.has_tile_to_flip(move, direction):
                    return True
        return False

    def is_valid_coord(self, row:int, col:int)-> bool:
        ''' Method: is_valid_coord
            Parameters: self, row (integer), col (integer)
            Returns: boolean (True if row and col is valid, False otherwise)
            Does: Checks whether the given coordinate (row, col) is valid.
                  A valid coordinate must be in the range of the board.
        '''
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return True
        return False

    def has_tile_to_flip(self, move:tuple, direction:tuple):
        ''' Method: has_tile_to_flip
            Parameters: self, move (tuple), direction (tuple)
            Returns: boolean 
                     (True if there is any tile to flip, False otherwise)
            Does: Checks whether the player has any adversary's tile to flip
                  with the move they make.
                  About input: move is the (row, col) coordinate of where the 
                  player makes a move; direction is the direction in which the 
                  adversary's tile is to be flipped (direction is any tuple 
                  defined in MOVE_DIRS).
        '''
        i = 1
        if self.is_valid_coord(move[0], move[1]):
            while True:
                row = move[0] + direction[0] * i
                col = move[1] + direction[1] * i
                if not self.is_valid_coord(row, col) or self.board[row][col] == 0:
                    return False
                elif self.board[row][col] == self.current_player_mark:
                    break
                else:
                    i += 1
        return i > 1

    def get_legal_moves(self):
        ''' Method: get_legal_moves
            Parameters: self
            Returns: a list of legal moves that can be made
            Does: Finds all the legal moves the current player can make.
                  Every move is a tuple of coordinates (row, col).
        '''
        moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                move = (row, col)
                if self.is_legal_move(move):
                    moves.append(move)
        return moves

    def get_random_move(self):
        ''' Method: make_random_move
            Parameters: self
            Returns: nothing
            Does: Makes a random legal move on the board.
        '''
        moves = self.get_legal_moves()
        if moves:
            return choice(moves)
        return None
    

    def flip_tiles(self,move:tuple):
        ''' Method: flip_tiles
            Parameters: self
            Returns: nothing
            Does: Flips the adversary's tiles for current move. Also, 
                  updates the state of the board (1 for black tiles and 
                  2 for white tiles), increases the number of tiles of 
                  the current player by 1, and decreases the number of 
                  tiles of the adversary by 1.
        '''
        for direction in self.move_directions:
            if self.has_tile_to_flip(move, direction):
                i = 1
                while True:
                    row = move[0] + direction[0] * i
                    col = move[1] + direction[1] * i
                    if self.board[row][col] == self.current_player_mark:
                        break
                    else:
                        self.board[row][col] = self.current_player_mark
                        self.increase_curr_player_score()
                        self.decrease_adversary_score()
                        i += 1
    def reset(self):
        self.whtplayer = Player(self.whtplayer.mark)
        self.blkplayer = Player(self.blkplayer.mark)
        self.board = np.zeros((self.board_size,self.board_size))
        self.board[self.board_size//2-1,self.board_size//2-1] = self.whtplayer.mark
        self.board[self.board_size//2,self.board_size//2] = self.whtplayer.mark
        self.board[self.board_size//2-1,self.board_size//2] = self.blkplayer.mark
        self.board[self.board_size//2,self.board_size//2-1] = self.blkplayer.mark
        self.current_player_mark = self.blkplayer.mark
        self.current_score= 0

    def get_human_move(self):
        while True:
            move = input("ENTER YOUR MOVE (eg: x,y ): ")
            try:
                x,y = int(move.split(',')[0]), int(move.split(',')[1])
                if self.is_legal_move((x,y)):
                    print("correct move")
                    return (x,y)
                else:
                    print("illegal move")
            except ValueError:
                print('Incorrect move')

    def max_action_count(self) -> int:
        return self.board_size**2

    def state_description(self) -> np.ndarray:
        return tf.keras.utils.to_categorical(self.board, 3).flatten()

    def state_dim(self) -> int:
        return 192

    def is_game_over(self) -> bool:
        curr_player_can_play = self.has_legal_move()
        self.switch_player()
        adversary_can_play = self.has_legal_move()
        self.switch_player()

        if not curr_player_can_play or not adversary_can_play :
            return True
        else :
            return False

    def get_legal_move_id(self,move:tuple):
        for i in range(self.board_size**2):
            if i//self.board_size == move[0] and i%self.board_size == move[1]:
                return i
        return None
        

    def act_with_action_id(self, action_id: int):

        #Agent plays
        assert(not self.is_game_over())
        i = action_id // self.board_size
        j = action_id % self.board_size
        assert (self.is_legal_move((i, j)))
        self.apply_move((i,j))

        if self.is_game_over():
            if self.blkplayer.score <= self.whtplayer.score:
                self.current_score = -1.0
            else:
                self.current_score = 1.0
            return

        #random_player plays
        assert(not self.is_game_over())
        move = self.get_random_move()
        assert(self.is_legal_move(move))
        self.apply_move(move)

        if self.is_game_over():
            if self.blkplayer.score <= self.whtplayer.score:
                self.current_score = -1.0
            else:
                self.current_score = 1.0
            return


    def score(self) -> float:
        if self.is_game_over():
            if self.blkplayer.score <= self.whtplayer.score:
                return -10.0
            else:
                return 1.0
        else:
            return self.blkplayer.score
            

    def available_actions_ids(self) -> np.ndarray:

        legal_moves = self.get_legal_moves()
        lgl_moves_ids = []

        for move in legal_moves:
            lgl_moves_ids.append(self.get_legal_move_id(move))

        return np.array(lgl_moves_ids)

    def view(self):
        self.display_board()
        legal_moves = self.get_legal_moves()
        print(legal_moves)
        print(self.available_actions_ids())
        print("BLACK PLAYER SCORE : ",self.blkplayer.score)
        print("BLACK PLAYER SCORE : ",self.whtplayer.score)
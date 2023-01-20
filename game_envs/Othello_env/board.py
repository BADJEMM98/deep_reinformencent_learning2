import numpy as np

class Board():
    def __init__(self, wp_mark:int,blkp_mark:int,size:int=8):
        if size%2==0:
            self.size = size
            self.board = np.zeros(self.size,self.size)
            self.board[self.size/2-1,self.size/2-1] = wp_mark
            self.board[self.size/2,self.size/2] = wp_mark
            self.board[self.size/2-1,self.size/2] = blkp_mark
            self.board[self.size/2,self.size/2-1] = blkp_mark
        else:
            self.board = None

  


            


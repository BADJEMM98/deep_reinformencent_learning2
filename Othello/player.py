
class Player():
    def __init__(self,mark,is_current=False,is_random=True,is_human=False,is_agent=False):
        self.mark = mark
        self.score = 2
        self.is_random = is_random
        self.is_human = is_human
        self.is_agent = is_agent
        self.is_current = is_current

    
from deep_single_agent_env import DeepSingleAgentEnv
from othello_env import OthelloEnv
import keras
import numpy as np


def play_game(env:DeepSingleAgentEnv):
    env.reset()
    pi = keras.models.load_model("models\Othello\ppo_one_step_sctor_critic.h5")

    while not env.is_game_over():
        env.display_board()
        legal_moves = env.get_legal_moves()
        aa = env.available_actions_ids()
        mask = np.zeros((env.max_action_count(),))
        mask[aa] = 1.0
        s = env.state_description()
        pi_s = pi([np.array([s]), np.array([mask])])[0].numpy()

        allowed_pi_s = pi_s[aa]
        sum_allowed_pi_s = np.sum(allowed_pi_s)
        if sum_allowed_pi_s == 0.0:
            probs = np.ones((len(aa),)) * 1.0 / (len(aa))
        else:
            probs = allowed_pi_s / sum_allowed_pi_s

        action_id = np.random.choice(aa, p=probs)
    
        assert(not env.is_game_over())
        i = action_id // env.board_size
        j = action_id % env.board_size
        assert (env.is_legal_move((i, j)))
        env.apply_move((i,j))

        env.current_score = 0
        
        
        if env.is_game_over():
            if env.blkplayer.score <= env.whtplayer.score:
                env.current_score = -1
            else:
                env.current_score = 1
            break

        print(legal_moves)

        move = env.get_human_move()
        env.apply_move(move)

        print("selected move is : ",move)

        print("BLACK PLAYER SCORE : ",env.blkplayer.score)
        print("BLACK PLAYER SCORE : ",env.whtplayer[0].score)

if __name__ == "__main__":

    play_game(OthelloEnv())
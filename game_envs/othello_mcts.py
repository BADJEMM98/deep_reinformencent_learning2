from deep_single_agent_env import DeepSingleAgentEnv
from othello_env import OthelloEnv
from tqdm import tqdm
import os
import math
import numpy as np

def monte_carlo_tree_search_and_choose_action(env: DeepSingleAgentEnv,
                                              iteration_count: int = 200) -> int:
    tree = {}

    root = env.state_id()
    tree[root] = {}
    for a in env.available_actions_ids():
        tree[root][a] = {
            'mean_score': 0.0,
            'selection_count': 0,
            'consideration_count': 0,
        }

    for _ in range(iteration_count):
        cloned_env = env.clone()
        current_node = cloned_env.state_id()

        nodes_and_chosen_actions = []

        # SELECTION
        while not cloned_env.is_game_over() and \
                not any(filter(lambda stats: stats['selection_count'] == 0, tree[current_node].values())):

            best_action = None
            best_action_score = None
            for (a, a_stats) in tree[current_node].items():
                ucb1_score = a_stats['mean_score'] + math.sqrt(2) * math.sqrt(
                    math.log(a_stats['consideration_count']) / a_stats['selection_count'])
                if best_action_score is None or ucb1_score > best_action_score:
                    best_action = a
                    best_action_score = ucb1_score

            nodes_and_chosen_actions.append((current_node, best_action))
            cloned_env.act_with_action_id(best_action)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = {
                        'mean_score': 0.0,
                        'selection_count': 0,
                        'consideration_count': 0,
                    }

        # EXPAND
        if not cloned_env.is_game_over():
            random_action = np.random.choice(list(
                map(lambda action_and_stats: action_and_stats[0],
                    filter(lambda action_and_stats: action_and_stats[1]['selection_count'] == 0,
                           tree[current_node].items())
                    )
            ))

            nodes_and_chosen_actions.append((current_node, random_action))
            cloned_env.act_with_action_id(random_action)
            current_node = cloned_env.state_id()

            if current_node not in tree:
                tree[current_node] = {}
                for a in cloned_env.available_actions_ids():
                    tree[current_node][a] = {
                        'mean_score': 0.0,
                        'selection_count': 0,
                        'consideration_count': 0,
                    }

        # EVALUATE / ROLLOUT
        while not cloned_env.is_game_over():
            cloned_env.act_with_action_id(np.random.choice(cloned_env.available_actions_ids()))

        score = cloned_env.score()

        # BACKUP / BACKPROPAGATE / UPDATE STATS
        for (node, chose_action) in nodes_and_chosen_actions:
            for a in tree[node].keys():
                tree[node][a]['consideration_count'] += 1
            tree[node][chose_action]['mean_score'] = (
                    (tree[node][chose_action]['mean_score'] * tree[node][chose_action]['selection_count'] + score) /
                    (tree[node][chose_action]['selection_count'] + 1)
            )
            tree[node][chose_action]['selection_count'] += 1

    most_selected_action = None
    most_selected_action_selection_count = None

    for (a, a_stats) in tree[root].items():
        if most_selected_action_selection_count is None or a_stats[
            'selection_count'] > most_selected_action_selection_count:
            most_selected_action = a
            most_selected_action_selection_count = a_stats['selection_count']

    return most_selected_action

def run_ttt_n_games_and_return_mean_score(games_count: int) -> float:
    env = OthelloEnv()
    total = 0.0
    wins = 0
    losses = 0
    draws = 0
    for _ in tqdm(range(games_count)):
        env.reset()

        while not env.is_game_over():
            chosen_a = monte_carlo_tree_search_and_choose_action(env)
            env.act_with_action_id(chosen_a)

        if env.score() > 0:
            wins += 1
        elif env.score() < 0:
            losses += 1
        else:
            draws += 1
        total += env.score()

    print(f"MCTS - wins : {wins}, losses : {losses}, draws : {draws}")
    print(f"MCTS - mean_score : {total / games_count}")
    return total / games_count


if __name__ == "__main__":
    run_ttt_n_games_and_return_mean_score(1000)
# deep_reinformencent_learning2

Environnement de départ :
- Line World
- Grid World
- Cant Stop / Push your luck
 > https://fr.wikipedia.org/wiki/Can%27t_Stop_(jeu)#R.C3.A8gle_du_jeu
 > https://boardgamearena.com/gamepanel?game=cantstop
 > version Android gratuite : https://play.google.com/store/apps/details?id=uk.co.mento.
dicefree&hl=en_US&gl=US
Environnements au choix :
- Bomberman (non discrétisé)
- PacMan (avec fantômes) (non discrétisé)
- Othello versus Random
- X (choix des étudiants si validé par l'enseignant)
Types d'agents à étudier :
- Random
- TabularQLearning (quand possible)
- DeepQLearning
- DoubleDeepQLearning
- DoubleDeepQLearningWithExperienceReplay
- DoubleDeepQLearningWithPrioritizedExperienceReplay
- REINFORCE
- REINFORCE with mean baseline
- REINFORCE with Baseline Learned by a Critic
- PPO A2C style
- RandomRollout
- Monte Carlo Tree Search (UCT)
- Expert Apprentice
- Alpha Zero
- MuZero
- MuZero stochastique
Metrics à obtenir (attention métriques pour la policy obtenue, pas pour la policy en mode entrainement) :
- Score moyen (pour chaque agent) au bout de 1000 parties
- Score moyen (pour chaque agent) au bout de 10 000 parties
- Score moyen (pour chaque agent) au bout de 100 000 parties
- Score moyen (pour chaque agent) au bout de 1 000 000 parties (si possible)
- Score moyen (pour chaque agent) au bout de XXX parties (si possible)
- Temps moyen mis pour exécuter un coup
- Longueur moyenne (nombre de step) d'une partie au bout de 1000 parties
- Longueur moyenne (nombre de step) d'une partie au bout de 10 000 parties
- Longueur moyenne (nombre de step) d'une partie au bout de 100 000 parties
- Longueur moyenne (nombre de step) d'une partie au bout de 1 000 000 parties (si possible)
- Longueur moyenne d'une partie au bout de XXX parties (si possible)
Il sera également nécessaire de présenter une interface graphique permettant de regarder jouer chaque
agent et également de mettre à disposition un agent 'humain'.
Pour chaque environnement, les étudiants devront étudier les performances de l'agent ou de chaque
couple d'agents possible et retranscrire leur résultats.
Les étudiants devront fournir l'intégralité du code leur ayant permis d'obtenir leurs résultats ainsi que les
modèles (keras/tensorflow/pytorch/???) entraînés et sauvegardés prêts à être exécutés pour confirmer
les résultats présentés.
Les é tudiants devront pré senter ces ré sultats dans un rapport ainsi qu'une pré sentation. Dans ces
derniers, les étudiants devront faire valoir leur méthodologie de choix d'hyperparamètres, et proposer leur
interprétation des résultats obtenus
# Board Games with Reinforcement Learning

## Structure of git
Notebooks for training are in base folder.
Sub folders are:  
- envs: implementations of different board games as OpenAI Gym environments.
- agents: wrappers around KerasRL agents that allow for multiplayer games.


## Implemented Games:
- Can't Stop (DQN Agents)

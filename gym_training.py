from rlxevolution import RLxEvolution


if __name__ == "__main__":
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    print('yay')
    env_name = 'BipedalWalker-v3'
    #env_name = "CartPole-v1"
    #env_name = 'LunarLanderContinuous-v2'
    agent = RLxEvolution(env_name, 10)
    agent.run_evolution() # train as PPO
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    #agent.evaluate(agent.actors[0], 10)
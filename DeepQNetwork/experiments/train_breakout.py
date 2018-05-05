import gym


from DeepQNetwork import algo


def main():
    # Make the environment
    env = gym.make('Breakout-v0')
    model = algo.models.mlp([64])
    act = algo.learn(

    )



if __name__ == '__main__':
    main()


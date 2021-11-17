from rllib_integration.carla_env import CarlaEnv
from sys import getsizeof
import os
import yaml
from dqn_example.dqn_experiment import DQNExperiment
import matplotlib.pyplot as plt
from rllib_integration.carla_core import kill_all_servers

EXPERIMENT_CLASS = DQNExperiment

if __name__=="__main__":
    try:
        os.environ["CARLA_ROOT"] = "/home/liquanyi/carla"
        with open("dqn_example/dqn_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["env_config"]["experiment"]["others"]["framestack"] = 3
        config["env_config"]["experiment"]["others"]["normalize"] = True
        config["env_config"]["experiment"]["others"]["gray_scale"] = True
        env = CarlaEnv(config["env_config"])
        env.reset()
        for i in range(10000):
            o,r,d,_ = env.step(8)
            plt.imshow(o)
            plt.savefig("obs_radius_{}.png".format(i))
            print("obs memory usage: {}".format(getsizeof(o)))

    finally:
        kill_all_servers()


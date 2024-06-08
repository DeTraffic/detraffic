import pathlib
import pickle

from game import Game
from matplotlib import pyplot as plt
from sumo_rl import (
    arterial4x4,
    cologne1,
    cologne3,
    cologne8,
    grid4x4,
    ingolstadt1,
    ingolstadt7,
    ingolstadt21,
)
from utils import load_yaml


class Experiment:
    def __init__(self, name, models, envs, iters, hyperparams):
        self.name: str = name
        self.models: list[pathlib.Path] = models
        self.envs: [str] = envs
        self.iters: [int] = iters
        self.hyperparams = hyperparams
        self.results = {}

    def run(self):
        for env_name in self.envs:
            match env_name:
                case "grid4x4":
                    env_cls = grid4x4
                case "arterial4x4":
                    env_cls = arterial4x4
                case "cologne1":
                    env_cls = cologne1
                case "cologne3":
                    env_cls = cologne3
                case "cologne8":
                    env_cls = cologne8
                case "ingolstadt1":
                    env_cls = ingolstadt1
                case "ingolstadt7":
                    env_cls = ingolstadt7
                case "ingolstadt21":
                    env_cls = ingolstadt21

            self.results[env_name] = {}

            env = env_cls(parallel=True, sumo_warnings=False)

            for model_conf in self.models:
                for iter_count in self.iters:
                    game = Game(env, env_name, model_conf, self.hyperparams, iter_count)
                    result = game.run()
                    self.results[env_name][model_conf.name] = result

        return self.results

    def save(self, path: pathlib.Path):
        with open(path, "wb") as f_obj:
            pickle.dump(
                self.results,
                f_obj,
                protocol=None,
                fix_imports=True,
                buffer_callback=None,
            )

    @staticmethod
    def load(path: pathlib.Path):
        with open(path, "rb") as f_obj:
            result = pickle.load(f_obj)
        return result

    @staticmethod
    def plot(path: pathlib.Path, results: dict, iter_count: int):
        for env_name, models in results.items():
            plt.figure(figsize=(16, 7))
            for metric in (
                "cumulative_reward",
                "system_total_stopped",
                "system_total_waiting_time",
                "system_mean_waiting_time",
                "system_mean_speed",
            ):
                plt.title(env_name)
                plt.xlabel("time_steps")
                plt.ylabel(metric)
                for model, metrics in models.items():
                    plt.plot(
                        range(len(metrics[metric])),
                        metrics[metric],
                        label=model.split(".")[0],
                    )
                for i in range(iter_count + 1):
                    plt.axvline((i) * 720, linestyle="dashed")
                plt.legend()
                plt.savefig(path / f"{env_name}_{metric}.png")
                plt.clf()

    @classmethod
    def from_dict(cls, config_dict: dict):
        name = config_dict["name"]
        models = [
            pathlib.Path(model)
            if model.endswith((".yaml", ".yml"))
            else pathlib.Path(f"{model}.yaml")
            for model in config_dict["models"]
        ]
        envs = config_dict["envs"]
        iters = config_dict["iters"]
        hyperparams = config_dict["hyperparams"]

        return cls(
            name=name, models=models, envs=envs, iters=iters, hyperparams=hyperparams
        )

    @classmethod
    def from_config(cls, config_path: pathlib.Path):
        config_dict = load_yaml(config_path)
        return cls.from_dict(config_dict)

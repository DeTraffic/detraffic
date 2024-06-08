"""summary."""

import math
import pathlib
import random
from collections import deque, namedtuple

import torch
from model_builder import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    """summary."""

    def __init__(self, capacity):
        """Summary."""
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    """summary."""

    def __init__(
        self, policy_net, target_net, optimizer, criterion, memory, hyperparams
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = memory
        self.hyperparams = hyperparams

    def select_action(self, state, steps_done):
        sample = random.random()
        eps_threshold = self.hyperparams["eps_end"] + (
            self.hyperparams["eps_start"] - self.hyperparams["eps_end"]
        ) * math.exp(-1.0 * steps_done / self.hyperparams["eps_decay"])
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1).item()
        else:
            return torch.tensor(
                [
                    [
                        random.sample(
                            [
                                i
                                for i in range(
                                    self.policy_net.sequential_layer[-1].out_features
                                )
                            ],
                            k=1,
                        )
                    ]
                ],
                device=device,
                dtype=torch.long,
            ).item()

    def optimize_policy(self):
        if len(self.memory) < self.hyperparams["batch_size"]:
            return

        transitions = self.memory.sample(self.hyperparams["batch_size"])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.tensor(batch.state).to(device)
        action_batch = torch.tensor(batch.action).to(device)
        reward_batch = torch.tensor(batch.reward).to(device)

        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        next_state_values = torch.zeros(self.hyperparams["batch_size"], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (
            next_state_values * self.hyperparams["gamma"]
        ) + reward_batch

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def optimize_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.hyperparams[
                "tau"
            ] + target_net_state_dict[key] * (1 - self.hyperparams["tau"])
        self.target_net.load_state_dict(target_net_state_dict)

    def remember_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    @classmethod
    def from_config(
        cls,
        n_observations: int,
        n_actions: int,
        model_conf: pathlib.Path,
        hyperparams: dict,
    ):
        """Summary."""
        policy_net = Model.from_config(
            n_observations=n_observations, n_actions=n_actions, config_path=model_conf
        ).to(device)

        target_net = Model.from_config(
            n_observations=n_observations, n_actions=n_actions, config_path=model_conf
        ).to(device)

        target_net.load_state_dict(policy_net.state_dict())

        optimizer = hyperparams["optimizer"]

        match optimizer:
            case "adam":
                optimizer = torch.optim.Adam

        optimizer = optimizer(policy_net.parameters())

        criterion = hyperparams["criterion"]

        match criterion:
            case "smooth_l1_loss":
                criterion = torch.nn.SmoothL1Loss

        criterion = criterion()

        replay_memory_capacity = hyperparams["replay_memory_capacity"]

        memory = ReplayMemory(replay_memory_capacity)

        return cls(policy_net, target_net, optimizer, criterion, memory, hyperparams)

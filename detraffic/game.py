import time

import torch
import tqdm
from agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Game:
    def __init__(self, env, env_name, model_conf, hyperparams, iter_count):
        self.env = env
        self.env_name = env_name
        self.model_conf = model_conf
        self.hyperparams = hyperparams
        self.iter_count = iter_count

        self._init()

    def _init(self):
        self.env.reset()

        self.agents = {}

        for agent in self.env.agents:
            n_observations = self.env.observation_space(agent).shape[0]
            n_actions = self.env.action_space(agent).n

            self.agents[agent] = Agent.from_config(
                n_observations=n_observations,
                n_actions=n_actions,
                model_conf=self.model_conf,
                hyperparams=self.hyperparams,
            )

    def run(self):
        steps_done = 0
        cumulative_reward = 0
        metrics_to_track = {
            "cumulative_reward",
            "system_total_stopped",
            "system_total_waiting_time",
            "system_mean_waiting_time",
            "system_mean_speed",
        }
        metrics = {metric: 720 * self.iter_count * [0] for metric in metrics_to_track}
        start_time = time.perf_counter()

        for i_episode in tqdm.tqdm(
            range(self.iter_count), desc=f"{self.env_name}: {self.model_conf.name}"
        ):
            # Initialize the environment and get its state
            states, infos = self.env.reset()
            states = {
                agent_id: torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                for (agent_id, state) in states.items()
            }

            pbar = tqdm.tqdm(total=3600)

            while self.env.agents:
                actions = self._select_actions(states, steps_done)
                states, rewards, terminations, truncations, infos = self.env.step(
                    actions
                )
                rewards = self._get_rewards(rewards)
                dones = self._get_dones(terminations, truncations)

                next_states = self._get_next_states(terminations, states)

                # Store the transition in memory
                self._remember_transitions(states, actions, next_states, rewards)

                # Move to the next state
                states = next_states

                # Perform one step of the optimization (on the policy network)
                self._optimize_policies()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                self._optimize_targets()

                total_reward = self._total_reward(rewards)
                cumulative_reward += total_reward
                metrics["cumulative_reward"][steps_done] = cumulative_reward

                for agent_id in self.env.agents:
                    metrics["system_total_stopped"][steps_done] = infos[agent_id][
                        "system_total_stopped"
                    ]
                    metrics["system_total_waiting_time"][steps_done] = infos[agent_id][
                        "system_total_waiting_time"
                    ]
                    metrics["system_mean_waiting_time"][steps_done] = infos[agent_id][
                        "system_mean_waiting_time"
                    ]
                    metrics["system_mean_speed"][steps_done] = infos[agent_id][
                        "system_mean_speed"
                    ]
                    break

                steps_done += 1
                pbar.update(5)
                pbar.set_description(f"Cumulative reward: {cumulative_reward:.2f}")
            pbar.close()

        stop_time = time.perf_counter()

        metrics["start_time"] = start_time
        metrics["stop_time"] = stop_time
        metrics["process_time"] = stop_time - start_time

        return metrics

    def _select_actions(self, states: dict, steps_done: int):
        return {
            agent_id: self.agents[agent_id].select_action(states[agent_id], steps_done)
            for (agent_id, state) in states.items()
        }

    def _get_rewards(self, rewards):
        return {
            agent_id: torch.tensor([reward], device=device)
            for (agent_id, reward) in rewards.items()
        }

    def _get_dones(self, terminations: dict, truncations: dict):
        return {
            agent_id: (terminations[agent_id] or truncations[agent_id])
            for agent_id in terminations
        }

    def _get_next_states(self, terminations, states):
        next_states = {}

        for agent_id, termination in terminations.items():
            if termination:
                next_states[agent_id] = None

            else:
                next_states[agent_id] = torch.tensor(
                    states[agent_id], dtype=torch.float32, device=device
                ).unsqueeze(0)

        return next_states

    def _remember_transitions(self, states, actions, next_states, rewards):
        for agent_id, agent in self.agents.items():
            agent.remember_transition(
                states[agent_id],
                actions[agent_id],
                next_states[agent_id],
                rewards[agent_id],
            )

    def _optimize_policies(self):
        for agent_id, agent in self.agents.items():
            agent.optimize_policy()

    def _optimize_targets(self):
        for agent_id, agent in self.agents.items():
            agent.optimize_target()

    def _total_reward(self, rewards):
        total_reward = 0

        for _, reward in rewards.items():
            total_reward += reward

        return total_reward.item()

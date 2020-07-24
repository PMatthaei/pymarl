from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
from absl import logging

from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from smac.env import StarCraft2Env

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

agent_reward_behaviour = {
    "selfish": .0,
    "selfless": 1.0,
    "balanced": .5,
}


class CombinedRewardsSMAC(StarCraft2Env):
    def __init__(self, map_name="8m", step_mul=8, move_amount=2, difficulty="7", game_version=None, seed=None,
                 continuing_episode=False, obs_all_health=True, obs_own_health=True, obs_last_action=False,
                 obs_pathing_grid=False, obs_terrain_height=False, obs_instead_of_state=False,
                 obs_timestep_number=False, state_last_action=True, state_timestep_number=False,
                 reward_sparse=False,
                 reward_only_positive=True,
                 reward_death_value=10,
                 reward_win=200,
                 reward_defeat=0,
                 reward_negative_scale=0.5,
                 reward_scale=True,
                 reward_scale_rate=20,
                 reward_local=True,
                 reward_local_weighted=True,
                 local_reward_weight=.5,
                 debug_rewards=False,
                 replay_dir="", replay_prefix="",
                 window_size_x=1920, window_size_y=1200, heuristic_ai=False, heuristic_rest=False, debug=False):
        """
                Create a modified SMAC environment which supports global, local and combined rewards

                Parameters (only recently introduced parameters. For a list of previous parameters see original SMAC)
                ----------
                debug_rewards : bool, optional
                    Debug reward process for each agent in each step.

                reward_local : bool, optional
                    Activate local reward calculation

                reward_local_weighted : bool, optional
                    Activate combined local rewards. The weighting is set via local_reward_weight.

                local_reward_weight : float, optional
                    The combination/weighting factor to combine local and global reward signals.
                """
        super().__init__(map_name, step_mul, move_amount, difficulty, game_version, seed, continuing_episode,
                         obs_all_health, obs_own_health, obs_last_action, obs_pathing_grid, obs_terrain_height,
                         obs_instead_of_state, obs_timestep_number, state_last_action, state_timestep_number,
                         reward_sparse, reward_only_positive, reward_death_value, reward_win, reward_defeat,
                         reward_negative_scale, reward_scale, reward_scale_rate, replay_dir, replay_prefix,
                         window_size_x, window_size_y, heuristic_ai, heuristic_rest, debug)

        self.debug_rewards = debug_rewards
        self.reward_local = reward_local
        self.reward_local_weighted = reward_local_weighted
        # Every agent receives same combination weight
        self.local_reward_weights = [local_reward_weight] * self.n_agents

        self.local_attack_r_t = 0

    def step(self, actions):

        if self.debug:
            logging.debug("New step = {}".format(self._episode_steps).center(60, '-'))

        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        # Collect attacked units aka targets in this step
        targets = []

        # Let AI/Agent decide on a action
        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action, target_id = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num, target_id = self.get_agent_action_heuristic(a_id, action)
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

            # Save action target for each agent
            if self.reward_local:
                targets.append(target_id)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return [0] * self.n_agents, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        local_rewards = []
        reward = 0
        if self.reward_local:
            # Calculate total attack reward based on targets attacked in step t - before rewarding !
            self.local_attack_r_t = self.calculate_local_attack_reward(targets)

            # Calculate for each agent its local reward
            for a_id, action in enumerate(actions_int):
                target_id = targets[a_id]
                local_reward = self.local_reward(a_id, target_id)
                local_rewards.append(local_reward)

            # Weight reward importance
            if self.reward_local_weighted:
                local_rewards = self.combine_local_rewards(local_rewards)

            # Calculate global reward to assert correctness later
            reward = self.global_reward()
        else:
            reward = self.global_reward()

        # After(!) rewarding every unit -> mark dead units for next steps
        self.mark_dead_units()

        terminated = False

        info = {"battle_won": False}

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    if self.debug_rewards:
                        logging.debug("Reward win with".format(self.reward_win).center(60, '-'))
                    reward += self.reward_win
                    if self.reward_local:
                        # Every agent receives win reward
                        local_reward_win = self.reward_win / self.n_agents
                        if self.debug_rewards:
                            logging.debug("Reward win locally with {}".format(local_reward_win).center(60, '-'))
                        local_rewards = [x + local_reward_win for x in local_rewards]
                else:
                    reward = 1
                    if self.reward_local:
                        local_reward_win = 1.0 / self.n_agents
                        if self.debug_rewards:
                            logging.debug("Reward win locally with {}".format(local_reward_win).center(60, '-'))
                        local_rewards = [local_reward_win] * self.n_agents

            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    if self.debug_rewards:
                        logging.debug("Reward loss with {}".format(self.reward_defeat).center(60, '-'))
                    reward += self.reward_defeat
                    if self.reward_local:
                        # Every agent receives win reward
                        local_reward_defeat = self.reward_defeat / self.n_agents
                        if self.debug_rewards:
                            logging.debug("Reward loss locally with {}".format(local_reward_defeat).center(60, '-'))
                        local_rewards = [x + local_reward_defeat for x in local_rewards]
                else:
                    reward = -1
                    if self.reward_local:
                        local_reward_defeat = -1.0 / self.n_agents
                        logging.debug("Reward local loss with {}".format(local_reward_defeat).center(60, '-'))
                        local_rewards = [local_reward_defeat] * self.n_agents

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug_rewards or self.debug:
            if self.reward_local:
                logging.debug("Local rewards = {}".format(local_rewards).center(60, '-'))
                logging.debug("Reward = {}".format(np.sum(local_rewards)).center(60, '-'))

            else:
                logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate
            if self.reward_local:
                local_rewards = [r / (self.max_reward / self.reward_scale_rate) for r in local_rewards]

        # Assert correctness of local reward function -> local reward sum == global reward
        if self.reward_local:
            local_reward_mean = np.mean(local_rewards)
            diff = abs(reward - local_reward_mean)

            np.testing.assert_almost_equal(reward, local_reward_mean, decimal=10,
                                           err_msg="Global reward and local reward mean should be equal. Difference = {}"
                                           .format(diff).center(60, '-'))

            if self.debug_rewards:
                logging.debug("Difference global vs. local = {}".format(diff).center(60, '-'))
                logging.debug("Local reward mean = {}".format(local_reward_mean).center(60, '-'))

        if self.debug_rewards:
            logging.debug("Global Reward = {}".format(reward).center(60, '-'))

        if self.reward_local:
            assert isinstance(local_rewards, list)  # Ensure reward is a list
            return local_rewards, terminated, info

        reward_filled = [reward] * self.n_agents  # Serve global reward to all agents
        assert isinstance(reward_filled, list)  # Ensure reward is a list
        return reward_filled, terminated, info

    def combine_local_rewards(self, rs):
        ws = self.local_reward_weights
        if self.debug:
            logging.debug("Local reward weights = {}".format(self.local_reward_weights).center(60, '-'))
        rs_ws = list(zip(rs, ws))  # Pair local rewards with their weight
        r_mean_weighted = np.mean([(1.0 - w_j) * r_j for r_j, w_j in rs_ws])
        rs_weighted = [w_i * r_i + r_mean_weighted for r_i, w_i in rs_ws]

        rs_sum = np.sum(rs)
        rsw_sum = np.sum(rs_weighted)
        diff = abs(rs_sum - rsw_sum)
        # Ensure weighting/combining returns almost same global reward
        np.testing.assert_almost_equal(rs_sum, rsw_sum, decimal=10,
                                       err_msg="Weighted reward sum must (almost) be equal to local reward sum. Difference = {}"
                                       .format(diff))
        return rs_weighted

    def local_reward(self, a_id, target_id):
        """
        Reward function which returns the local reward of a single agent with a given id.

        The agent receives the following positive rewards:
            - his health delta compared to previous step -> healing or shield regeneration
            - local attack reward (see: calculate_local_attack_reward() )
        The agent receives the following negative rewards:
            - his health delta compared to previous step -> received damage
            - death penalty
        """
        if self.reward_sparse:
            return 0

        neg_scale = self.reward_negative_scale

        delta_death = 0
        delta_self = 0
        delta_enemy = 0

        agent_unit = self.get_unit_by_id(a_id)

        # If the unit with id a_id is still alive
        if not self.death_tracker_ally[a_id]:
            # Fetch its previous health
            prev_health = self.previous_ally_units[a_id].health + self.previous_ally_units[a_id].shield
            # If it just died
            if agent_unit.health == 0:
                if self.debug_rewards:
                    logging.debug("Agent with id {} died.".format(a_id).center(60, '-'))
                # Reward death negatively if configured
                if not self.reward_only_positive:
                    delta_death -= self.reward_death_value * neg_scale
                # Remember lost health (= damage from enemies) to reward negatively later
                delta_self += prev_health * neg_scale
            # If still alive
            else:
                current_health = agent_unit.health + agent_unit.shield
                health_reward = prev_health - current_health
                # Reward the delta health (shield regeneration, heal through MMM)
                delta_self += neg_scale * health_reward

        if self.debug_rewards:
            logging.debug("Local death penalty for {} = {}".format(a_id, delta_death).center(60, '-'))
            logging.debug("Local damage taken by {} = {}".format(a_id, delta_self).center(60, '-'))

        # Search for the target (if it exists) which a_id attacked
        if target_id is not None:
            e_id, e_unit = next((e_id, _) for e_id, _ in self.enemies.items() if target_id == e_id)
            if not self.death_tracker_enemy[e_id]:
                # Reward dealt attack damage + kill
                delta_enemy += self.local_attack_r_t
                if self.debug_rewards:
                    logging.debug("Local damage reward for {} = {}".format(a_id, self.local_attack_r_t).center(60, '-'))

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_death)
        else:
            reward = delta_enemy + delta_death - delta_self

        return reward

    def global_reward(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # health rewards update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    if not self.reward_only_positive:
                        scaled_death_reward = self.reward_death_value * neg_scale
                        delta_deaths -= scaled_death_reward
                        if self.debug_rewards:
                            logging.debug("Reward global death penalty {}".format(scaled_death_reward).center(60, '-'))

                    scaled_health_reward = prev_health * neg_scale
                    delta_ally += scaled_health_reward
                    if self.debug_rewards:
                        logging.debug("Adding to global taken damage = {}".format(scaled_health_reward).center(60, '-'))
                else:
                    # still alive
                    scaled_health_reward = neg_scale * (prev_health - al_unit.health - al_unit.shield)
                    delta_ally += scaled_health_reward
                    if self.debug_rewards:
                        logging.debug("Adding to global taken damage = {}".format(scaled_health_reward).center(60, '-'))

        # attack rewards
        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = self.previous_enemy_units[e_id].health + self.previous_enemy_units[e_id].shield

                if e_unit.health == 0:
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                    if self.debug_rewards:
                        logging.debug("Enemy with id {} died.".format(e_id).center(60, '-'))
                        logging.debug("Reward enemy death = {}".format(self.reward_death_value).center(60, '-'))
                        logging.debug("Reward enemy damage = {}".format(prev_health).center(60, '-'))
                else:
                    health_reward = prev_health - e_unit.health - e_unit.shield
                    delta_enemy += health_reward
                    if self.debug_rewards:
                        logging.debug("Reward enemy damage = {}".format(health_reward).center(60, '-'))

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        return reward / self.n_agents

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (target is None or self.agents[target].health == 0 or
                    self.agents[target].health == self.agents[target].health_max):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (al_unit.health != 0 and
                            al_unit.health != al_unit.health_max):
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             al_unit.pos.x, al_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['heal']
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (unit.unit_type == self.marauder_id and
                            e_unit.unit_type == self.medivac_id):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             e_unit.pos.x, e_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['attack']
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if (self.heuristic_rest and
                self.get_avail_agent_actions(a_id)[action_num] == 0):

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y)
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y)
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount)
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount)
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions['move'],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False)
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert avail_actions[action] == 1, \
            "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        target_id = None

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None, target_id
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, target_id

    def mark_dead_units(self):
        # mark allies
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id] and al_unit.health == 0:
                self.death_tracker_ally[al_id] = 1
        # mark enemies
        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id] and e_unit.health == 0:
                self.death_tracker_enemy[e_id] = 1

    def calculate_local_attack_reward(self, a_id_targets):
        # Retrieve total dmg dealt by the team of agent
        total_attack_reward = self.get_total_attack_dmg()
        # Calculate amount of attacking agents
        attackers = sum(x is not None for x in a_id_targets)
        # Each attacking agent receives an equal portion of the total dealt damage
        # Since SC2 is deterministic in its damage calculation
        # we can assume the same dealt damage by same units for each step -> split reward evenly
        # This may not be applicable to heterogeneous unit groups
        return total_attack_reward / (1.0 if (attackers == 0) else float(attackers))

    def get_total_attack_dmg(self):
        total_attack_reward = 0
        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = self.previous_enemy_units[e_id].health + self.previous_enemy_units[e_id].shield
                if e_unit.health == 0:
                    if self.debug_rewards:
                        logging.debug("Enemy with id {} died.".format(e_id).center(60, '-'))
                    # Reward kill for a_id
                    total_attack_reward += self.reward_death_value
                    # Reward the damage which led to the death
                    total_attack_reward += prev_health

                    if self.debug_rewards:
                        logging.debug("Reward enemy death = {}".format(self.reward_death_value).center(60, '-'))
                        logging.debug("Reward enemy damage = {}".format(prev_health).center(60, '-'))
                else:
                    # Reward the health difference - assuming units do not regenerate health/shield
                    health_delta = prev_health - (e_unit.health + e_unit.shield)
                    assert health_delta >= 0, \
                        "Enemy unit health delta is negative. " \
                        "This unit {} gained health since last step and can thus not be counted as damage reward" \
                            .format(e_id).center(60, '-')
                    total_attack_reward += health_delta

                    if self.debug_rewards:
                        logging.debug("Reward enemy damage = {}".format(health_delta).center(60, '-'))
        return total_attack_reward

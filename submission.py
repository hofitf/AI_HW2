from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


def min_distance_charge(env: WarehouseEnv, robot_id: int):
    mister_robot = env.get_robot(robot_id)
    if manhattan_distance(env.charge_stations[0].position, mister_robot.position) < manhattan_distance(
            env.charge_stations[1].position, mister_robot.position):
        return manhattan_distance(env.charge_stations[0].position, mister_robot.position)
    return manhattan_distance(env.charge_stations[1].position, mister_robot.position)


def best_credit_package_distance(env: WarehouseEnv, robot_id: int):
    mister_robot = env.get_robot(robot_id)
    md_0 = manhattan_distance(env.packages[0].position, env.packages[0].destination)
    md_1 = manhattan_distance(env.packages[1].position, env.packages[1].destination)
    if md_0 > md_1:
        return env.packages[0], manhattan_distance(mister_robot.position, env.packages[0].position), md_0, 1
    return env.packages[1], manhattan_distance(mister_robot.position, env.packages[1].position), md_1, 0


def second_credit_package_distance(env: WarehouseEnv, robot_id: int, package_id: int):
    mister_robot = env.get_robot(robot_id)
    other_package = env.packages[package_id]
    md_package = manhattan_distance(other_package.position, other_package.destination)
    md_robot = manhattan_distance(mister_robot.position, other_package.position)
    return md_robot, md_package


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    mister_robot = env.get_robot(robot_id)
    other_robot = env.get_robot(1)
    if robot_id == 1 :
        other_robot = env.get_robot(0)
    best_credit = best_credit_package_distance(env, robot_id)
    other_distance = second_credit_package_distance(env, robot_id, best_credit[3])
    if mister_robot.package is not None and mister_robot.battery > manhattan_distance(mister_robot.position,
                                                                                      mister_robot.package.destination):
        return 500 * manhattan_distance(mister_robot.package.position, mister_robot.package.destination) - \
            manhattan_distance(mister_robot.position, mister_robot.package.destination) + \
            500 * mister_robot.credit + 100 * mister_robot.battery
    elif mister_robot.package is not None and mister_robot.battery <= \
            manhattan_distance(mister_robot.position, mister_robot.package.destination):
        return 500 * mister_robot.credit + 100 * mister_robot.battery - min_distance_charge(env, robot_id)
    elif mister_robot.package is None and (best_credit[1] + best_credit[2] + 2) < mister_robot.battery and \
            best_credit[0].on_board:
        return 500 * mister_robot.credit + 50 * 2 * best_credit[2] - best_credit[1]
    elif mister_robot.package is None and (other_distance[1] + other_distance[0] + 2) < mister_robot.battery \
            and env.packages[best_credit[3]].on_board:
        return 500 * mister_robot.credit + 20 * 2 * other_distance[1] - other_distance[0]
    elif mister_robot.credit <= other_robot.credit:
        1000 * mister_robot.credit + 1000 * mister_robot.battery - min_distance_charge(env, robot_id)
    else:
        return 500 * mister_robot.credit + 500 * mister_robot.battery - min_distance_charge(env, robot_id)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)

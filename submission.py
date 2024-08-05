from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
from func_timeout import func_timeout, FunctionTimedOut


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
    other_robot = env.get_robot((robot_id + 1) % 2)
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
    elif mister_robot.credit < other_robot.credit:
        return 1000 * mister_robot.credit - 100 * min_distance_charge(env, robot_id)
    else:
        return 500 * mister_robot.credit - min_distance_charge(env, robot_id)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):

    def RB_Minimax(self, env: WarehouseEnv, agent_id, depth, turn):
        if depth == 0:
            return smart_heuristic(env, agent_id)
        if turn % 2 == 0:
            max_value = float('-inf')
            operators, children = self.successors(env, agent_id)
            for child in children:
                value = self.RB_Minimax(child, agent_id, depth - 1, turn + 1)
                max_value = max(max_value, value)
            return max_value
        else:
            min_value = float('inf')
            other_agent = (agent_id + 1) % 2
            operators, children = self.successors(env, other_agent)
            for child in children:
                value = self.RB_Minimax(child, agent_id, depth - 1, turn + 1)
                min_value = min(min_value, value)
            return min_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_move = None
        best_value = float('-inf')
        depth = 1
        try:
            while True:
                current_best_value = float('-inf')
                current_best_move = None
                operators, children = self.successors(env, agent_id)
                for child, op in zip(children, operators):
                    move_value = func_timeout(time_limit - (time.time() - start_time) - 0.1, self.RB_Minimax,
                                              args=(child, agent_id, depth, 1))
                    if move_value > current_best_value:
                        current_best_value = move_value
                        current_best_move = op

                if current_best_value > best_value:
                    best_value = current_best_value
                    best_move = current_best_move

                depth += 1
        except FunctionTimedOut:
            return best_move


class AgentAlphaBeta(Agent):
    def RB_alpha_beta(self, env: WarehouseEnv, agent_id, depth, turn, alpha, beta):
        if depth == 0:
            return smart_heuristic(env, agent_id)
        if turn % 2 == 0:
            max_value = float('-inf')
            operators, children = self.successors(env, agent_id)
            for child in children:
                value = self.RB_alpha_beta(child, agent_id, depth - 1, turn + 1, alpha, beta)
                max_value = max(max_value, value)
                alpha = max(max_value, alpha)
                if max_value >= beta:
                    return float('inf')
            return max_value
        else:
            min_value = float('inf')
            other_agent = (agent_id + 1) % 2
            operators, children = self.successors(env, other_agent)
            for child in children:
                value = self.RB_alpha_beta(child, agent_id, depth - 1, turn + 1, alpha, beta)
                min_value = min(min_value, value)
                beta = min(min_value, beta)
                if min_value <= alpha:
                    return float('-inf')
            return min_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_move = None
        best_value = float('-inf')
        depth = 1
        try:
            while True:
                current_best_value = float('-inf')
                current_best_move = None
                operators, children = self.successors(env, agent_id)
                for child, op in zip(children, operators):
                    move_value = func_timeout(time_limit - (time.time() - start_time) - 0.1, self.RB_alpha_beta,
                                              args=(child, agent_id, depth, 1, float('-inf'), float('inf')))
                    if move_value > current_best_value:
                        current_best_value = move_value
                        current_best_move = op

                if current_best_value > best_value:
                    best_value = current_best_value
                    best_move = current_best_move

                depth += 1
        except FunctionTimedOut:
            return best_move


class AgentExpectimax(Agent):

    def RB_Expectimax(self, env: WarehouseEnv, agent_id, depth, turn):
        if depth == 0:
            return smart_heuristic(env, agent_id)
        if turn % 2 == 1:
            other_agent = (agent_id + 1) % 2
            operators, children = self.successors(env, other_agent)
            num_2 = 0
            if "pick_up" in operators:
                num_2 += 1
            if "move right" in operators:
                num_2 += 1
            sum_prob = 0
            if num_2 == 0:
                for child in children:
                    sum_prob += (1 / len(operators)) * self.RB_Expectimax(child, agent_id, depth - 1, turn + 1)
            elif num_2 == 1:
                for child, op in zip(children, operators):
                    if op == "pick_up" or op == "move right":
                        sum_prob += 2 * (1 / (len(operators) + 2)) * self.RB_Expectimax(child, agent_id, depth - 1,
                                                                                        turn + 1)
                    else:
                        sum_prob += (1 / (len(operators) + 2)) * self.RB_Expectimax(child, agent_id, depth - 1,
                                                                                    turn + 1)
            else:
                for child, op in zip(children, operators):
                    if op == "pick_up" or op == "move right":
                        sum_prob += 2 * (1 / (len(operators) + 4)) * self.RB_Expectimax(child, agent_id, depth - 1,
                                                                                        turn + 1)
                    else:
                        sum_prob += (1 / (len(operators) + 4)) * self.RB_Expectimax(child, agent_id, depth - 1,
                                                                                    turn + 1)
            return sum_prob
        if turn % 2 == 0:
            max_value = float('-inf')
            operators, children = self.successors(env, agent_id)
            for child in children:
                value = self.RB_Expectimax(child, agent_id, depth - 1, turn + 1)
                max_value = max(max_value, value)
            return max_value
        else:
            min_value = float('inf')
            other_agent = (agent_id + 1) % 2
            operators, children = self.successors(env, other_agent)
            for child in children:
                value = self.RB_Expectimax(child, agent_id, depth - 1, turn + 1)
                min_value = min(min_value, value)
            return min_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_move = None
        best_value = float('-inf')
        depth = 1
        try:
            while True:
                current_best_value = float('-inf')
                current_best_move = None
                operators, children = self.successors(env, agent_id)
                for child, op in zip(children, operators):
                    move_value = func_timeout(time_limit - (time.time() - start_time) - 0.1, self.RB_Expectimax,
                                              args=(child, agent_id, depth, 1))
                    if move_value > current_best_value:
                        current_best_value = move_value
                        current_best_move = op

                if current_best_value > best_value:
                    best_value = current_best_value
                    best_move = current_best_move

                depth += 1
        except FunctionTimedOut:
            return best_move


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

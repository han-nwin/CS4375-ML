import random
import time
from collections import defaultdict

# ----------------------------
# Terminal colors (ANSI)
# ----------------------------
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"


def ctext(s: str, color: str) -> str:
    return f"{color}{s}{RESET}"


# ----------------------------
# Grid World with Apples
# ----------------------------
EMPTY = "."
WALL = "#"
ROBOT = "R"
GOOD = "G"  # good apple
BAD = "B"  # bad apple

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

DIRS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "STAY": (0, 0),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class AppleWorld:
    def __init__(
        self,
        h=25,
        w=50,
        wall_prob=0.12,
        seed=0,
        spawn_good=(1, 3),  # spawn between 1..3 good apples each step
        spawn_bad=(1, 2),  # spawn between 1..2 bad apples each step
        apple_ttl=(8, 16),  # each apple lives between 8..16 steps
    ):
        random.seed(seed)
        self.h = h
        self.w = w
        self.spawn_good = spawn_good
        self.spawn_bad = spawn_bad
        self.apple_ttl = apple_ttl

        # Static walls
        self.walls = [[False for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                # border walls
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    self.walls[r][c] = True
                else:
                    self.walls[r][c] = random.random() < wall_prob

        # Ensure a roomy center + a few open spots
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if (r in (h // 2, h // 2 - 1)) and (c in (w // 2, w // 2 - 1)):
                    self.walls[r][c] = False

        self.reset()

    def reset(self):
        # Apples: dict[(r,c)] = (type, ttl)
        self.apples = {}
        self.t = 0
        self.good_eaten = 0
        self.bad_eaten = 0

        # Place robot in an empty cell
        self.robot = self._random_empty_cell()
        self.robot_dir = "RIGHT"  # initial direction
        return self.observe()

    def _random_empty_cell(self):
        while True:
            r = random.randint(1, self.h - 2)
            c = random.randint(1, self.w - 2)
            if not self.walls[r][c] and (r, c) not in self.apples:
                return (r, c)

    def _spawn_apples(self):
        # Spawn multiple apples per step
        g = random.randint(self.spawn_good[0], self.spawn_good[1])
        b = random.randint(self.spawn_bad[0], self.spawn_bad[1])

        for _ in range(g):
            self._spawn_one(GOOD)

        for _ in range(b):
            self._spawn_one(BAD)

    def _spawn_one(self, kind):
        # Try a few times to find an empty cell
        for _ in range(20):
            r, c = self._random_empty_cell()
            # don't spawn on robot
            if (r, c) == self.robot:
                continue
            ttl = random.randint(self.apple_ttl[0], self.apple_ttl[1])
            self.apples[(r, c)] = (kind, ttl)
            return

    def _decay_apples(self):
        # Decrease TTL; remove expired
        to_del = []
        for pos, (kind, ttl) in self.apples.items():
            ttl -= 1
            if ttl <= 0:
                to_del.append(pos)
            else:
                self.apples[pos] = (kind, ttl)
        for pos in to_del:
            del self.apples[pos]

    def cell_type(self, r, c):
        if self.walls[r][c]:
            return WALL
        if (r, c) in self.apples:
            kind, _ttl = self.apples[(r, c)]
            return kind
        return EMPTY

    def observe(self):
        """
        Small "local observation" state:
        - what's in current cell (EMPTY/GOOD/BAD)
        - what's in 4 neighbors (WALL/EMPTY/GOOD/BAD)
        This keeps Q-table small and learnable.
        """
        r, c = self.robot
        cur = self.cell_type(r, c)
        up = self.cell_type(r - 1, c)
        down = self.cell_type(r + 1, c)
        left = self.cell_type(r, c - 1)
        right = self.cell_type(r, c + 1)
        return (cur, up, down, left, right)

    def step(self, action):
        """
        Rewards:
        - step cost: -0.05 (encourage efficiency)
        - bump wall: -0.75
        - move to GOOD apple: +10 (auto-eat)
        - move to BAD apple: -10 (auto-eat)
        """
        self.t += 1
        reward = -0.05

        r, c = self.robot

        if action in ("UP", "DOWN", "LEFT", "RIGHT", "STAY"):
            dr, dc = DIRS[action]
            nr, nc = r + dr, c + dc
            if self.walls[nr][nc]:
                reward -= 0.75
            else:
                self.robot = (nr, nc)
                # Update direction based on movement
                if action in ("UP", "DOWN", "LEFT", "RIGHT"):
                    self.robot_dir = action

                # Auto-eat apple if present at new position
                if self.robot in self.apples:
                    kind, _ttl = self.apples[self.robot]
                    del self.apples[self.robot]
                    if kind == GOOD:
                        reward += 10.0
                        self.good_eaten += 1
                    else:
                        reward -= 10.0
                        self.bad_eaten += 1

        # Environment dynamics after action
        self._decay_apples()
        self._spawn_apples()

        obs = self.observe()
        done = False  # continuous task
        return obs, reward, done

    def render(self):
        rr, rc = self.robot
        # Map directions to arrow symbols
        arrow_map = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
        robot_symbol = arrow_map.get(self.robot_dir, "→")

        lines = []
        for r in range(self.h):
            row = []
            for c in range(self.w):
                if self.walls[r][c]:
                    row.append(ctext("#", WHITE))
                elif (r, c) == (rr, rc):
                    row.append(ctext(robot_symbol, YELLOW))
                elif (r, c) in self.apples:
                    kind, _ttl = self.apples[(r, c)]
                    if kind == GOOD:
                        row.append(ctext("G", GREEN))
                    else:
                        row.append(ctext("B", RED))
                else:
                    row.append(ctext(".", GRAY))
            lines.append("".join(row))
        return "\n".join(lines)


# ----------------------------
# Q-Learning Agent
# ----------------------------
class QAgent:
    def __init__(
        self,
        actions,
        alpha=0.3,
        gamma=0.90,
        epsilon=0.25,
        eps_decay=0.9995,
        eps_min=0.05,
    ):
        self.actions = actions
        self.Q = defaultdict(lambda: {a: 0.0 for a in actions})
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def act(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q = self.Q[state]
        best_a = max(q, key=q.get)
        return best_a

    def update(self, s, a, r, s2):
        q_sa = self.Q[s][a]
        best_next = max(self.Q[s2].values())
        target = r + self.gamma * best_next
        self.Q[s][a] = (1 - self.alpha) * q_sa + self.alpha * target

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ----------------------------
# Run / Train
# ----------------------------
def clear_screen():
    print("\033[2J\033[H", end="")


def run(
    steps=4000,
    render_every=20,
    sleep=0.25,
    seed=0,
):
    env = AppleWorld(seed=seed)
    agent = QAgent(ACTIONS, alpha=0.25, gamma=0.90, epsilon=0.35)

    state = env.reset()
    total = 0.0

    for t in range(1, steps + 1):
        action = agent.act(state)
        next_state, reward, _done = env.step(action)
        agent.update(state, action, reward, next_state)
        agent.decay_epsilon()

        state = next_state
        total += reward

        if t % render_every == 0:
            clear_screen()
            print(env.render())
            print()
            print(
                f"step={t}  total_reward={total:.1f}  "
                f"eps={agent.epsilon:.3f}  last_action={action}  last_reward={reward:+.2f}"
            )
            print(
                ctext("G=good apple (+10)", GREEN), " ", ctext("B=bad apple (-10)", RED)
            )
            print(
                f"Apples eaten - {ctext('Good: ' + str(env.good_eaten), GREEN)}  "
                f"{ctext('Bad: ' + str(env.bad_eaten), RED)}"
            )
            time.sleep(sleep)

    clear_screen()
    print(env.render())
    print("\nDone.")
    print(f"Final total reward: {total:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(
        f"Total apples eaten - {ctext('Good: ' + str(env.good_eaten), GREEN)}  "
        f"{ctext('Bad: ' + str(env.bad_eaten), RED)}"
    )


if __name__ == "__main__":
    run(
        steps=6000,  # train longer = better behavior
        render_every=0.25,  # lower = more frequent redraw
        sleep=0.1,  # animation speed
        seed=3,
    )

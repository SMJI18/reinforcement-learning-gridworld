

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import warnings
import os
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ─── MDP / Environment parameters ───────────────────────────────────────────
GRID_SIZE    = 4
GAMMA        = 0.99
THETA        = 1e-4          # VI/PI convergence threshold
ACTIONS      = [(-1,0),(1,0),(0,-1),(0,1)]   # Up Down Left Right
ACTION_SYM   = ['↑','↓','←','→']

REWARD_GOAL  =  1.0
REWARD_BOMB  = -1.0
REWARD_STEP  = -0.04

# Q-Learning hyper-parameters
QL_EPISODES   = 1000
QL_ALPHA      = 0.1
QL_EPS_START  = 0.2
QL_EPS_END    = 0.01
QL_MAX_STEPS  = 100
EVAL_EPISODES = 100

# ─── Four scenarios ──────────────────────────────────────────────────────────
def make_scenarios():
    S = {}

    # Scenario 1 – Single Wall
    S[1] = dict(
        name="Scenario 1: Single Wall",
        walls={(1,1)},
        bombs=set(),
        goal=(3,3),
        start=(0,0)
    )

    # Scenario 2 – Walls + Bomb
    S[2] = dict(
        name="Scenario 2: Walls + Bomb",
        walls={(1,1),(2,1)},
        bombs={(0,2)},
        goal=(3,3),
        start=(0,0)
    )

    # Scenario 3 – Corner Bombs (bombs flank goal, causing risk-averse goal avoidance)
    S[3] = dict(
        name="Scenario 3: Corner Bombs",
        walls=set(),
        bombs={(2,3),(3,2)},
        goal=(3,3),
        start=(0,0)
    )

    # Scenario 4 – Narrow Passage
    S[4] = dict(
        name="Scenario 4: Narrow Passage",
        walls={(1,0),(1,1),(1,2),(1,3),(2,0),(2,2),(2,3)},
        bombs={(3,1)},
        goal=(3,3),
        start=(0,0)
    )

    return S


# ─── Gridworld environment ────────────────────────────────────────────────────
class Gridworld:
    def __init__(self, scenario):
        self.walls  = scenario['walls']
        self.bombs  = scenario['bombs']
        self.goal   = scenario['goal']
        self.start  = scenario['start']
        self.n      = GRID_SIZE
        self.states = [(r,c) for r in range(self.n) for c in range(self.n)
                       if (r,c) not in self.walls]

    def is_terminal(self, s):
        return s == self.goal or s in self.bombs

    def reward(self, s):
        if s == self.goal:  return REWARD_GOAL
        if s in self.bombs: return REWARD_BOMB
        return REWARD_STEP

    def step(self, s, a):
        if self.is_terminal(s):
            return s, self.reward(s)
        dr, dc = ACTIONS[a]
        ns = (s[0]+dr, s[1]+dc)
        # boundary / wall → stay
        if not (0 <= ns[0] < self.n and 0 <= ns[1] < self.n) or ns in self.walls:
            ns = s
        return ns, self.reward(ns)

    def reset(self):
        return self.start


# ─── Value Iteration ──────────────────────────────────────────────────────────
def value_iteration(env):
    V     = {s: 0.0 for s in env.states}
    deltas = []

    while True:
        delta = 0
        for s in env.states:
            if env.is_terminal(s):
                V[s] = env.reward(s)
                continue
            v = V[s]
            V[s] = max(
                env.reward(ns) + GAMMA * V.get(ns, 0)
                for a in range(4)
                for ns, _ in [env.step(s, a)]
            )
            delta = max(delta, abs(v - V[s]))
        deltas.append(delta)
        if delta < THETA:
            break

    # extract greedy policy
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = None
            continue
        best_a, best_v = 0, -1e9
        for a in range(4):
            ns, _ = env.step(s, a)
            val = env.reward(ns) + GAMMA * V.get(ns, 0)
            if val > best_v:
                best_v, best_a = val, a
        policy[s] = best_a

    return V, policy, deltas


# ─── Policy Iteration ─────────────────────────────────────────────────────────
def policy_iteration(env):
    policy = {s: 0 for s in env.states}
    V      = {s: 0.0 for s in env.states}
    eval_errors = []

    while True:
        # --- policy evaluation ---
        while True:
            delta = 0
            for s in env.states:
                if env.is_terminal(s):
                    V[s] = env.reward(s)
                    continue
                v = V[s]
                ns, _ = env.step(s, policy[s])
                V[s]  = env.reward(ns) + GAMMA * V.get(ns, 0)
                delta = max(delta, abs(v - V[s]))
            eval_errors.append(delta)
            if delta < THETA:
                break

        # --- policy improvement ---
        stable = True
        for s in env.states:
            if env.is_terminal(s):
                continue
            old = policy[s]
            best_a, best_v = 0, -1e9
            for a in range(4):
                ns, _ = env.step(s, a)
                val   = env.reward(ns) + GAMMA * V.get(ns, 0)
                if val > best_v:
                    best_v, best_a = val, a
            policy[s] = best_a
            if old != policy[s]:
                stable = False

        if stable:
            break

    return V, policy, eval_errors


# ─── Q-Learning ───────────────────────────────────────────────────────────────
def q_learning(env, seed=SEED):
    rng = np.random.default_rng(seed)
    Q   = {(s, a): 0.0 for s in env.states for a in range(4)}
    returns = []

    for ep in range(QL_EPISODES):
        eps = QL_EPS_START + (QL_EPS_END - QL_EPS_START) * ep / QL_EPISODES
        s   = env.reset()
        ep_ret = 0

        for _ in range(QL_MAX_STEPS):
            if env.is_terminal(s):
                break
            if rng.random() < eps:
                a = rng.integers(4)
            else:
                a = max(range(4), key=lambda x: Q[(s, x)])

            ns, r = env.step(s, a)
            best_next = max(Q[(ns, x)] for x in range(4)) if not env.is_terminal(ns) else 0
            Q[(s, a)] += QL_ALPHA * (r + GAMMA * best_next - Q[(s, a)])
            ep_ret += r
            s = ns

        returns.append(ep_ret)

    # extract greedy policy & V
    policy = {}
    V      = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = None
            V[s]      = env.reward(s)
        else:
            best_a    = max(range(4), key=lambda x: Q[(s, x)])
            policy[s] = best_a
            V[s]      = Q[(s, best_a)]

    return V, policy, returns


# ─── Evaluation: goal-reach % and avg return ──────────────────────────────────
def evaluate(env, policy, seed=SEED, n_ep=EVAL_EPISODES):
    rng = np.random.default_rng(seed)
    goal_reach = 0
    ep_returns = []

    for _ in range(n_ep):
        s      = env.reset()
        ep_ret = 0
        reached = False
        for _ in range(QL_MAX_STEPS):
            if env.is_terminal(s):
                if s == env.goal:
                    reached = True
                break
            a      = policy[s] if policy[s] is not None else 0
            s, r   = env.step(s, a)
            ep_ret += r
        goal_reach += int(reached)
        ep_returns.append(ep_ret)

    return goal_reach / n_ep * 100, np.mean(ep_returns), np.std(ep_returns)


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def plot_heatmap(ax, env, V, title):
    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for s, v in V.items():
        grid[s[0], s[1]] = v

    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    vcenter    = 0 if vmin < 0 < vmax else (vmin + vmax) / 2
    norm       = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    im = ax.imshow(grid, cmap='RdYlGn', norm=norm, origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # overlay special cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = (r, c)
            if s in env.walls:
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#333333'))
            elif s in env.bombs:
                ax.text(c, r, '✕', ha='center', va='center',
                        fontsize=14, color='red', fontweight='bold')
            elif s == env.goal:
                ax.text(c, r, '★', ha='center', va='center',
                        fontsize=14, color='gold', fontweight='bold')
            elif not np.isnan(grid[r, c]):
                ax.text(c, r, f'{grid[r,c]:.2f}', ha='center', va='center',
                        fontsize=7, color='black')

    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(GRID_SIZE)); ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE)); ax.set_yticklabels(range(GRID_SIZE))
    ax.set_xlabel('Column'); ax.set_ylabel('Row')


def plot_policy(ax, env, policy, title):
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_facecolor('#f8f8f8')

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = (r, c)
            if s in env.walls:
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#444'))
            elif s in env.bombs:
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#fee2e2'))
                ax.text(c, r, '✕', ha='center', va='center',
                        fontsize=14, color='red', fontweight='bold')
            elif s == env.goal:
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color='#fef9c3'))
                ax.text(c, r, '★', ha='center', va='center',
                        fontsize=14, color='#ca8a04', fontweight='bold')
            elif s in policy and policy[s] is not None:
                ax.text(c, r, ACTION_SYM[policy[s]], ha='center', va='center',
                        fontsize=16, color='#1e3a5f')

    for i in range(GRID_SIZE+1):
        ax.axhline(i-.5, color='#ccc', lw=0.8)
        ax.axvline(i-.5, color='#ccc', lw=0.8)

    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(GRID_SIZE)); ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE)); ax.set_yticklabels(range(GRID_SIZE))
    ax.set_xlabel('Column'); ax.set_ylabel('Row')


def plot_convergence(ax, data, title, ylabel, color='steelblue'):
    ax.semilogy(data, color=color, lw=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    scenarios = make_scenarios()

    print(f"{'Scenario':<28} {'Algo':>4}  {'Goal%':>6}  {'Avg Return':>22}")
    print("─" * 70)

    for sid, scenario in scenarios.items():
        env = Gridworld(scenario)

        # Run all three algorithms
        V_vi,  pi_vi,  deltas_vi  = value_iteration(env)
        V_pi,  pi_pi,  errs_pi    = policy_iteration(env)
        V_ql,  pi_ql,  returns_ql = q_learning(env)

        # Evaluate
        gr_vi, mu_vi, std_vi = evaluate(env, pi_vi)
        gr_pi, mu_pi, std_pi = evaluate(env, pi_pi)
        gr_ql, mu_ql, std_ql = evaluate(env, pi_ql)

        name = scenario['name']
        print(f"{name:<28} {'VI':>4}  {gr_vi:>5.0f}%  {mu_vi:>8.3f} ± {std_vi:.3f}")
        print(f"{'':28} {'PI':>4}  {gr_pi:>5.0f}%  {mu_pi:>8.3f} ± {std_pi:.3f}")
        print(f"{'':28} {'QL':>4}  {gr_ql:>5.0f}%  {mu_ql:>8.3f} ± {std_ql:.3f}")
        print()

        # ── Figure: 2 rows × 3 columns ──
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(scenario['name'], fontsize=13, fontweight='bold', y=1.01)

        plot_heatmap(axes[0,0], env, V_vi, 'VI Value Heatmap')
        plot_policy (axes[0,1], env, pi_vi, 'Greedy Policy (VI)')
        plot_convergence(axes[0,2], deltas_vi,
                         'Convergence of Value Iteration',
                         'Max |ΔV|', color='steelblue')
        plot_convergence(axes[1,0], errs_pi,
                         'Convergence of Policy Iteration\n(Eval Error)',
                         'Max |ΔV|', color='darkorange')
        # Q-Learning rolling mean for readability
        window = max(1, len(returns_ql)//20)
        smooth = np.convolve(returns_ql, np.ones(window)/window, mode='valid')
        axes[1,1].plot(smooth, color='cornflowerblue', lw=1.0, alpha=0.9)
        axes[1,1].set_title('Q-Learning Episodic Return', fontsize=9)
        axes[1,1].set_xlabel('Episodes'); axes[1,1].set_ylabel('Return')
        axes[1,1].grid(True, alpha=0.3)

        # Summary table in last subplot
        axes[1,2].axis('off')
        table_data = [
            ['', 'Goal%', 'Avg Return'],
            ['VI', f'{gr_vi:.0f}%', f'{mu_vi:.3f}±{std_vi:.3f}'],
            ['PI', f'{gr_pi:.0f}%', f'{mu_pi:.3f}±{std_pi:.3f}'],
            ['QL', f'{gr_ql:.0f}%', f'{mu_ql:.3f}±{std_ql:.3f}'],
        ]
        tbl = axes[1,2].table(cellText=table_data, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.8)
        axes[1,2].set_title('Evaluation Summary', fontsize=9)

        plt.tight_layout()
        # ensure a local outputs directory next to the script
        outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        fname = os.path.join(outputs_dir, f'scenario_{sid}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → saved {fname}")

    # ── Combined summary table ──
    print("\n=== Goal Reach % and Avg Episodic Return ===")
    print(f"{'Scenario':<22} {'VI Goal':>8} {'PI Goal':>8} {'QL Goal':>8}"
          f"  {'VI Return':>16} {'PI Return':>16} {'QL Return':>16}")
    for sid, scenario in scenarios.items():
        env = Gridworld(scenario)
        V_vi,  pi_vi,  _  = value_iteration(env)
        V_pi,  pi_pi,  _  = policy_iteration(env)
        V_ql,  pi_ql,  _  = q_learning(env)
        gr_vi, mu_vi, std_vi = evaluate(env, pi_vi)
        gr_pi, mu_pi, std_pi = evaluate(env, pi_pi)
        gr_ql, mu_ql, std_ql = evaluate(env, pi_ql)
        print(f"{scenario['name']:<22} {gr_vi:>7.0f}% {gr_pi:>7.0f}% {gr_ql:>7.0f}%"
              f"  {mu_vi:>7.3f}±{std_vi:.3f}  {mu_pi:>7.3f}±{std_pi:.3f}  {mu_ql:>7.3f}±{std_ql:.3f}")


if __name__ == '__main__':
    main()

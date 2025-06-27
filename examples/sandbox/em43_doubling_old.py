"""
em43_numba.py  -  Numba-accelerated batched EM-4/3 simulator
==================================================================
Drop-in replacement for the original **em43_parallel.py**.  
Public API unchanged:

    from em43_numba import EM43Batch, _sanitize_rule, _sanitize_programme

Key details
-----------
* 1-D CA, 4 states, radius-1, open boundary, 2-cell separator “BB”.
* Evaluates **B inputs in parallel** for a single genome.
* Core simulation loop is compiled with Numba (`@njit(cache=True)`).
* First call takes a few 100 ms to compile, then runs 5-10x faster.

No bit-packing; all arrays are `uint8`.  First & last columns stay blank.

Author: Giacomo Bocchese - with the help of ChatGPT

this code has not been checked - may still present unexpected behaviours
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import numba as nb

nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)   # use all available
print(f"Numba using {nb.get_num_threads()} threads")


# ────────────────── helpers & constants ──────────────────────────────
def lut_idx(l: int, c: int, r: int) -> int:              # 3-tuple → 0..63
    return (l << 4) | (c << 2) | r


SEPARATOR = np.array([3, 3], dtype=np.uint8)             # BB

_IMMUTABLE = {                                           # hard-wired LUT rows
    lut_idx(0, 0, 0): 0,
    lut_idx(0, 2, 0): 2,
    lut_idx(0, 0, 2): 0,
    lut_idx(2, 0, 0): 0,
    lut_idx(0, 3, 3): 3,
    lut_idx(3, 3, 0): 3,
    lut_idx(0, 0, 3): 0,
    lut_idx(3, 0, 0): 0,
}

def _sanitize_rule(rule: np.ndarray) -> np.ndarray:
    """Overwrite immutable LUT entries; clip to 0-3."""
    rule = rule.astype(np.uint8, copy=True)
    for k, v in _IMMUTABLE.items():
        rule[k] = v
    rule[rule > 3] &= 3
    return rule

def _sanitize_programme(prog: np.ndarray) -> np.ndarray:
    """Remove accidental blue cells from programme."""
    prog = prog.astype(np.uint8, copy=True)
    prog[prog == 3] = 0
    return prog


# ────────────────── Numba simulation kernel ──────────────────────────
@nb.njit(cache=True)
def _simulate(rule: np.ndarray,
              prog: np.ndarray,
              inputs: np.ndarray,
              window: int,
              max_steps: int,
              halt_th: float) -> np.ndarray:
    """
    Parameters
    ----------
    rule    : (64,) uint8
    prog    : (L,)  uint8
    inputs  : (B,) int64     (values 1..30)
    Returns
    -------
    outputs : (B,) int32     (-10 on failure)
    """
    L = prog.shape[0]
    B = inputs.shape[0]
    N = window

    state   = np.zeros((B, N), np.uint8)
    halted  = np.zeros(B, np.bool_)
    frozen  = np.zeros_like(state)
    output  = np.full(B, -10, np.int32)

    # write programme & separator
    for b in range(B):
        for j in range(L):
            state[b, j] = prog[j]
        state[b, L    ] = 3     # B
        state[b, L + 1] = 3     # B

    # write beacons 0^(n+1) R 0
    for b in range(B):
        r_idx = L + 2 + inputs[b] + 1
        state[b, r_idx] = 2

    # main loop
    for _ in range(max_steps):
        active_any = False
        for b in range(B):
            if halted[b]:
                continue
            active_any = True
            nxt = np.zeros(N, np.uint8)
            for x in range(1, N - 1):
                idx = (state[b, x-1] << 4) | (state[b, x] << 2) | state[b, x+1]
                nxt[x] = rule[idx]
            state[b] = nxt

            # halting check
            live = blue = 0
            for x in range(N):
                v = nxt[x]
                if v != 0:
                    live += 1
                    if v == 3:
                        blue += 1
            if live > 0 and blue / live >= halt_th:
                halted[b] = True
                frozen[b] = nxt

        if not active_any:
            break

    # decode outputs
    for b in range(B):
        if not halted[b]:
            continue
        rpos = -1
        for x in range(N - 1, -1, -1):
            if frozen[b, x] == 2:
                rpos = x
                break
        if rpos != -1:
            output[b] = rpos - (L + 3)          # (sep=2)+1 zeros before R

    return output

# Additional imports for GA
import math, pickle, time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ───────────────── Hyper-parameters ────────────────────────────────
POP_SIZE      = 100     # Number of genomes in the population (reduced for comparison)
N_GENERATIONS = 20       # Number of generations to run (reduced for comparison)
ELITE_FRAC    = 0.1      # Fraction of population to keep as elite
TOURNEY_K     = 3       # Number of genomes to select for tournament

P_MUT_RULE    = 0.03    # Probability of mutating each rule entry
P_MUT_PROG    = 0.08    # Probability of mutating each program cell
L_PROG        = 10      # Length of the program sequence

LAMBDA_P      = 0.01    # Sparsity penalty coefficient
EPS_RANDOM_IMMIGRANTS = 0.2  # Probability of introducing random immigrants
N_COMPLEX_TELEMETRY   = 30   # Frequency of detailed telemetry (in generations)

INPUT_SET   = np.arange(1, 31, dtype=np.int64)  # Input range: 1 to 30
TARGET_OUT  = 2 * INPUT_SET                    # Target output: 4x input
WINDOW      = 200      # Tape length for simulation
MAX_STEPS   = 800      # Maximum simulation steps
HALT_THRESH = 0.50     # Threshold for early stopping

CHECK_EVERY = 50      # Frequency of checkpoint saving
SAVE_DIR    = Path("dp_checkpoints")  # Directory for checkpoints
SAVE_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng()

# ───────────────── Vectorised fitness via Numba ────────────────────
@nb.njit(parallel=True, fastmath=True, cache=True)
def fitness_population(rules: np.ndarray, progs: np.ndarray) -> np.ndarray:
    """Compute fitness for every genome in parallel.
    rules  : (P,64)  uint8
    progs  : (P,L)   uint8
    returns: (P,)    float32
    """
    P = rules.shape[0]
    fitness = np.empty(P, dtype=np.float32)
    for i in nb.prange(P):
        outs = _simulate(rules[i], progs[i], INPUT_SET, WINDOW,
                          MAX_STEPS, HALT_THRESH)
        avg_err = np.abs(outs - TARGET_OUT).mean()
        sparsity = np.count_nonzero(progs[i]) / progs.shape[1]
        fitness[i] = -avg_err - LAMBDA_P * sparsity
    return fitness

# ───────────────── GA helpers ──────────────────────────────────────

def random_genome() -> tuple[np.ndarray, np.ndarray]:
    rule = rng.integers(0, 4, 64, dtype=np.uint8)
    prog = rng.choice([0, 1, 2], size=L_PROG, p=[0.7, 0.2, 0.1])
    return _sanitize_rule(rule), _sanitize_programme(prog)


def tournament(pop_rules, pop_progs, fit):
    idx = rng.choice(POP_SIZE, TOURNEY_K, replace=False)
    best = idx[np.argmax(fit[idx])]
    return pop_rules[best], pop_progs[best]


def crossover(rule1, prog1, rule2, prog2):
    vec1 = np.concatenate((rule1, prog1))
    vec2 = np.concatenate((rule2, prog2))
    cut  = rng.integers(1, vec1.size)
    child = np.concatenate((vec1[:cut], vec2[cut:]))
    return _sanitize_rule(child[:64]), _sanitize_programme(child[64:64+L_PROG])


def mutate(rule, prog):
    # LUT entries
    mask_r = rng.random(64) < P_MUT_RULE
    if mask_r.any():
        rule = rule.copy()
        rule[mask_r] = rng.integers(0, 4, mask_r.sum(), dtype=np.uint8)
        rule = _sanitize_rule(rule)
    # Programme cells
    mask_p = rng.random(L_PROG) < P_MUT_PROG
    if mask_p.any():
        prog = prog.copy()
        prog[mask_p] = rng.choice([0,1,2], size=mask_p.sum(), p=[0.7,0.2,0.1])
        prog = _sanitize_programme(prog)
    return rule, prog


def avg_pairwise_hamming(flat: np.ndarray) -> float:
    P = flat.shape[0]
    total = 0
    for i in range(P-1):
        diff = np.count_nonzero(flat[i+1:] != flat[i], axis=1)
        total += diff.sum()
    return total / (P*(P-1)//2)

# ───────────────── GA main loop ────────────────────────────────────

def run_ga():
    """
    Run Genetic Algorithm.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, float]
        (best_rule, best_prog, best_fitness)
    """
    # Population arrays
    pop_rules = np.empty((POP_SIZE, 64), np.uint8)
    pop_progs = np.empty((POP_SIZE, L_PROG), np.uint8)
    for i in range(POP_SIZE):
        r, p = random_genome()
        pop_rules[i], pop_progs[i] = r, p

    best_curve, mean_curve = [], []
    n_elite = int(math.ceil(ELITE_FRAC * POP_SIZE))
    n_imm   = max(1, int(EPS_RANDOM_IMMIGRANTS * POP_SIZE))

    for gen in tqdm(range(1, N_GENERATIONS+1), ncols=80, desc="GA"):
        fit = fitness_population(pop_rules, pop_progs)
        order = np.argsort(fit)[::-1]
        pop_rules, pop_progs, fit = pop_rules[order], pop_progs[order], fit[order]

        best_curve.append(float(fit[0]))
        mean_curve.append(float(fit.mean()))

        if gen % N_COMPLEX_TELEMETRY == 0:
            flat = np.concatenate((pop_rules, pop_progs), axis=1)
            ham = avg_pairwise_hamming(flat)
            tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}  ham={ham:.1f}")
        else:
            tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}")

        # Check-point
        if gen % CHECK_EVERY == 0 or gen == N_GENERATIONS:
            chk = {
                "gen": gen,
                "best_rule": pop_rules[0],
                "best_prog": pop_progs[0],
                "fit_best": float(fit[0]),
                "curve_best": best_curve,
                "curve_mean": mean_curve,
            }
            with open(SAVE_DIR / f"ckpt_gen{gen:04d}.pkl", "wb") as f:
                pickle.dump(chk, f)

        # ── produce next generation ──
        next_rules = pop_rules[:n_elite].copy()
        next_progs = pop_progs[:n_elite].copy()
        while next_rules.shape[0] < POP_SIZE:
            r1, p1 = tournament(pop_rules, pop_progs, fit)
            r2, p2 = tournament(pop_rules, pop_progs, fit)
            child_r, child_p = mutate(*crossover(r1, p1, r2, p2))
            next_rules = np.vstack((next_rules, child_r))
            next_progs = np.vstack((next_progs, child_p))

        # Random immigrants
        for _ in range(n_imm):
            idx = rng.integers(n_elite, POP_SIZE)
            next_rules[idx], next_progs[idx] = random_genome()

        pop_rules, pop_progs = next_rules, next_progs

    # # Save curves & best genome
    # plt.figure(figsize=(6,4))
    # plt.plot(best_curve, label="best"); plt.plot(mean_curve, label="mean")
    # plt.xlabel("generation"); plt.ylabel("fitness"); plt.legend(); plt.tight_layout()
    # plt.savefig("outputs/fitness_curve.png", dpi=150); plt.close()

    # with open("models/best_genome.pkl", "wb") as f:
    #     pickle.dump({"rule": pop_rules[0], "prog": pop_progs[0], "fitness": best_curve[-1]}, f)

# ───────────────── Entry point ─────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    run_ga()
    print(f"Elapsed {time.time()-t0:.1f}s")
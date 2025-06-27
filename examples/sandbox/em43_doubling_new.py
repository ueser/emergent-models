#!/usr/bin/env python3 
"""
EM43 with new SDK
"""

from emergent_models.core.state import StateModel
from emergent_models.rules.sanitization import lut_idx
from emergent_models.core.space_model import Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator
from emergent_models.training.new_fitness import AbsoluteDifferenceFitness
from emergent_models.training import (Trainer, AbsoluteDifferenceFitness, 
                                      ComplexityRewardFitness,
                    SparsityPenalizedFitness, GAOptimizer)
from emergent_models.training import TqdmMonitor, DetailedMonitor, CombinedMonitor

import numpy as np


print("🔬 Setting up investigation...")

# ═══════════════════ HYPERPARAMETERS ═══════════════════
POP_SIZE      = 100
N_GENERATIONS = 20
ELITE_FRAC    = 0.1
TOURNEY_K     = 3
P_MUT_RULE    = 0.03
P_MUT_PROG    = 0.08
L_PROG        = 10
LAMBDA_P      = 0.01
EPS_RANDOM_IMMIGRANTS = 0.2
N_COMPLEX_TELEMETRY   = 30
INPUT_SET     = np.arange(1, 31, dtype=np.int64)
TARGET_OUT    = 2 * INPUT_SET
WINDOW        = 200
MAX_STEPS     = 800
HALT_THRESH   = 0.50
CHECK_EVERY   = 50


# 1. Domain setup
_IMMUTABLE = {
    lut_idx(0, 0, 0): 0,  # Empty space stays empty
    lut_idx(0, 2, 0): 2,  # Red beacon propagation
    lut_idx(0, 0, 2): 0,  # Red beacon boundary
    lut_idx(2, 0, 0): 0,  # Red beacon boundary  
    lut_idx(0, 3, 3): 3,  # Blue boundary behavior
    lut_idx(3, 3, 0): 3,  # Blue boundary behavior
    lut_idx(0, 0, 3): 0,  # Blue boundary behavior
    lut_idx(3, 0, 0): 0,  # Blue boundary behavior
}
state = StateModel([0,1,2,3], immutable=_IMMUTABLE)
space = Tape1D(length=WINDOW, radius=1)
encoder = Em43Encoder(state, space)


# 2. Simulation
sim = Simulator(
    state=state, 
    space=space, 
    max_steps=MAX_STEPS,                                    # MAX_STEPS
    halt_thresh=HALT_THRESH                                 # HALT_THRESH
)

# Usage
base_fitness = AbsoluteDifferenceFitness(continuous=True)

fitness = ComplexityRewardFitness(
    base_fitness, 
    complexity_bonus=0.05
)


# Setup monitoring for CLI usage
# Force CLI tqdm (not notebook) for better terminal display
tqdm_monitor = TqdmMonitor(N_GENERATIONS, force_notebook=False)
detailed_monitor = DetailedMonitor(log_every=10, detailed_every=30)
combined = CombinedMonitor(tqdm_monitor, detailed_monitor)

# 4. Optimizer with all GA parameters
optim = GAOptimizer(
    pop_size=POP_SIZE,                                      # POP_SIZE
    state=state,
    prog_len=L_PROG,                                        # L_PROG
    mutation_rate=P_MUT_RULE,                               # P_MUT_RULE
    prog_mutation_rate=P_MUT_PROG,                          # P_MUT_PROG
    elite_fraction=ELITE_FRAC,                              # ELITE_FRAC
    tournament_size=TOURNEY_K,                              # TOURNEY_K
    random_immigrant_rate=EPS_RANDOM_IMMIGRANTS,            # EPS_RANDOM_IMMIGRANTS
    prog_sparsity=0.3
)

# 3. Training
trainer = Trainer(encoder, sim, fitness, optim, combined)



print("✅ Setup complete!")
print("🚀 Starting training with tqdm progress bar...")

# This should now show a nice CLI progress bar!
result = trainer.fit(
    inputs=np.arange(1, 31),  # Full input set (1-30)
    generations=N_GENERATIONS,  # Use full generations
    use_tqdm=False,          # Don't add another tqdm (we already have one in monitor)
    checkpoint_every=CHECK_EVERY,
    early_stopping_threshold=1.00
)

print(f"✅ Training completed! Best fitness: {result['best_fitness']:.4f}")
import numpy as np

from emergent_models.core.space import CASpace
from emergent_models.core.genome import Genome
from emergent_models.rules.base import RuleSet
from emergent_models.data.dataloader import CADataLoader
from emergent_models.data.dataset import CADataset
from emergent_models.simulators.base import Simulator
from emergent_models.losses.distance import HammingLoss


class IncrementDataset(CADataset):
    """Simple dataset of x -> x+1 for x in [1,32]."""

    def __len__(self) -> int:
        return 32

    def __getitem__(self, idx: int):
        x = idx + 1
        inp = CASpace(np.array([x], dtype=np.int32))
        target = CASpace(np.array([x + 1], dtype=np.int32))
        return inp, target


class IncrementRule(RuleSet):
    """Rule that maps each integer to another integer via a lookup table."""

    def __init__(self):
        super().__init__(neighborhood_size=1, n_states=64)
        # Initialize rule table randomly
        for i in range(64):
            self.set_rule((i,), np.random.randint(0, 64))

    def forward(self, space: CASpace) -> CASpace:
        data = np.array([
            self.rule_table.get((int(val),), int(val)) for val in space.data
        ], dtype=np.int32)
        return CASpace(data, space.device)


def evaluate(genome: Genome, dataloader: CADataLoader) -> float:
    loss_fn = HammingLoss()
    total = 0.0
    count = 0
    for batch in dataloader:
        for inp, target in batch:
            out = genome(inp)
            total += loss_fn(out, target)
            count += 1
    return total / count


def train_increment_rule(epochs: int = 200) -> Genome:
    dataset = IncrementDataset()
    dataloader = CADataLoader(dataset, batch_size=8, shuffle=True)
    rule = IncrementRule()
    genome = Genome(rule)

    current_loss = evaluate(genome, dataloader)
    for _ in range(epochs):
        # Randomly mutate one entry in the rule table
        idx = np.random.randint(0, 64)
        old_value = rule.rule_table[(idx,)]
        rule.rule_table[(idx,)] = np.random.randint(0, 64)
        new_loss = evaluate(genome, dataloader)
        if new_loss <= current_loss:
            current_loss = new_loss
        else:
            # Revert change if not improved
            rule.rule_table[(idx,)] = old_value
        if current_loss == 0:
            break
    return genome


def main() -> None:
    genome = train_increment_rule()
    simulator = Simulator(max_steps=1)
    dataset = IncrementDataset()
    print("Learned mapping:")
    for inp, target in dataset:
        out = simulator(genome, inp)
        print(f"{inp.data[0]} -> {out.data[0]} (target {target.data[0]})")


if __name__ == "__main__":
    main()

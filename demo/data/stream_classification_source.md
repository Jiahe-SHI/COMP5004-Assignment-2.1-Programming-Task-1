# Synthetic streaming classification data

Generated locally for this assignment using NumPy; no external dataset was used.

- Generator: numpy.random.default_rng(2026).
- Rows: 1200; numeric features: x1, x2, x3.
- Features: independent standard normal draws.
- Target: int((x1 > 0.25) or (x2 < -0.5)).
- Each target is independently flipped with probability 0.05.
- After target generation, each feature value is independently replaced with an empty field with probability 0.05.
- Random draws occur in this order: features, label flips, feature missingness.
- x3 is a noise feature; target values are never missing.
- CSV row order defines the simulated arrival order. There are no real timestamps or intentional concept drift.
- This dataset demonstrates functionality; results do not establish performance on real-world data.
- Generation environment: NumPy 1.26.4.

"""Logical NumPy array storage accounting for reachable state."""

import numpy as np


def array_storage_bytes(root):
    """Sum nbytes of distinct reachable ndarray objects.

    Traverse dictionaries, built-in containers and instance __dict__.
    Repeated references and cycles are visited once by object identity.
    Arrays are terminal objects: their base arrays and object elements
    are not traversed.

    Different array views are counted separately by logical nbytes.
    Python object overhead, object-array referents, slots, native
    allocations and temporary execution state are excluded.
    """
    pending = [root]
    seen = set()
    total = 0

    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)

        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif not isinstance(value, type) and not callable(value):
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                pending.append(attributes)

    return total
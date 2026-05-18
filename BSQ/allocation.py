import math

def allocate_bits(sensitivity, param_counts, budget, b_min=2, b_max=8):
    """
    Mixed precision allocator based on normalized sensitivity.
    """

    score = {}

    for k in sensitivity:
        score[k] = math.log(
            (sensitivity[k] + 1e-9) / (param_counts[k] + 1e-9)
        )

    s_min = min(score.values())
    s_max = max(score.values())

    bits = {}
    total = 0

    # initial allocation
    for k, v in score.items():
        b = b_min + (b_max - b_min) * (v - s_min) / (s_max - s_min + 1e-9)
        b = int(round(b))
        b = max(b_min, min(b_max, b))
        bits[k] = b
        total += b * param_counts[k]

    # greedy correction to respect budget
    while total > budget:
        worst = max(bits, key=lambda k: sensitivity[k] / (param_counts[k] + 1e-9))
        if bits[worst] > b_min:
            bits[worst] -= 1
            total -= param_counts[worst]
        else:
            break

    return bits
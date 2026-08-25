def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    S = len(values)
    A = len(rewards[0])

    new_values = []

    for s in range(S):
        best_value = float("-inf")

        for a in range(A):
            future_value = sum(
                transitions[s][a][next_s] * values[next_s]
                for next_s in range(S)
            )

            q_value = rewards[s][a] + gamma * future_value

            best_value = max(best_value, q_value)

        new_values.append(float(best_value))

    return new_values
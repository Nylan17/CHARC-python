import numpy as np

def generate_new_NARMA_sequence(sequence_length, memory_length, lower_dist=0, upper_dist=0.5):
    """
    Generates a sequence using a Nonlinear Autoregressive Moving Average (NARMA) model.
    """
    # Set default distribution if not provided
    if lower_dist is None:
        lower_dist = 0
    if upper_dist is None:
        upper_dist = 0.5

    # Initialize sequences
    washout = 1000
    total_length = sequence_length + memory_length + washout
    input_sequence = np.random.uniform(lower_dist, upper_dist, total_length)
    output_sequence = np.zeros(total_length)

    # NARMA model computation based on the memory length specified
    for i in range(memory_length, total_length):
        middle_sum = np.sum(output_sequence[i - memory_length + 1:i + 1])

        if memory_length in [10, 5, 20]:
            output_sequence[i] = (0.3 * output_sequence[i-1] +
                                  0.05 * output_sequence[i-1] * middle_sum +
                                  1.5 * input_sequence[i - memory_length] * input_sequence[i-1] +
                                  0.1)
        elif memory_length in [30, 40]:
            output_sequence[i] = (0.2 * output_sequence[i-1] +
                                  0.004 * output_sequence[i-1] * middle_sum +
                                  1.5 * input_sequence[i - memory_length] * input_sequence[i-1] +
                                  0.001)
        else:
            # This case handles other memory lengths generically
            for j in range(1, memory_length):
                output_sequence[i] += output_sequence[i-j]

            output_sequence[i] = (output_sequence[i] * 0.05 * output_sequence[i-1] +
                                  0.3 * output_sequence[i-1] +
                                  1.5 * input_sequence[i - memory_length] * input_sequence[i-1] +
                                  0.1)

    # Remove the initial transient part (washout)
    input_sequence = input_sequence[washout + memory_length:]
    output_sequence = output_sequence[washout + memory_length:]

    return input_sequence, output_sequence

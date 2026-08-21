"""
Fixed-point (Q(m, n)) conversion utilities.

Q(m, n) notation used in this project (matches the project specification):
    m = number of integer bits, INCLUDING the sign bit
    n = number of fractional bits
    total_bits = m + n

Example: Q(8, 8) is a 16-bit signed fixed-point format -- 8 bits for the
integer part (including sign) and 8 bits for the fractional part.

Conceptually, a fixed-point value stores a REAL number as an INTEGER that has
been multiplied ("scaled") by 2^n. To recover the real value, divide the
integer back by 2^n. Real hardware never performs true division -- it just
interprets the bit pattern differently, which is exactly what makes
fixed-point arithmetic cheap to implement (no floating-point unit needed).
"""


def scale_factor(qn):
    """2^n : the multiplier that converts a real number into a fixed-point integer."""
    return 1 << qn


def value_range(qm, qn):
    """
    Return (min_int, max_int): the smallest/largest INTEGER value that fits in
    a signed two's-complement number with (qm + qn) total bits.
    """
    total_bits = qm + qn
    max_int = (1 << (total_bits - 1)) - 1
    min_int = -(1 << (total_bits - 1))
    return min_int, max_int


def to_fixed(x, qm, qn):
    """
    Convert a real (float) number to its Q(qm, qn) fixed-point INTEGER
    representation, with rounding and saturation (clamping) on overflow.
    """
    scale = scale_factor(qn)
    min_int, max_int = value_range(qm, qn)
    q = round(x * scale)
    # Saturate instead of silently wrapping around -- well-designed hardware
    # does the same to avoid undetected overflow errors.
    return max(min_int, min(max_int, q))


def to_float(q, qn):
    """
    Convert a Q(*, qn) fixed-point integer back to an approximate float.
    Used only for debugging/reporting -- the hardware itself never does this.
    """
    return q / scale_factor(qn)


def to_hex(q, qm, qn):
    """
    Represent a signed fixed-point integer as an unsigned hexadecimal string,
    using two's-complement encoding, padded to ceil((qm+qn)/4) digits.
    This is the format you will paste into a VHDL ROM initialization table
    in Phase 3.
    """
    total_bits = qm + qn
    if q < 0:
        q = (1 << total_bits) + q  # two's complement encoding
    hex_digits = (total_bits + 3) // 4
    return format(q, f"0{hex_digits}X")


def fixed_point_layer(inputs_fixed, weights_fixed, biases_fixed, qm, qn, apply_relu):
    """
    Simulate one fully-connected layer using ONLY fixed-point integer
    arithmetic -- exactly the computation the VHDL MAC units + FSM will
    perform in Phase 3. Running the whole network through this function is
    what lets us measure, in Python, the accuracy the real hardware will get.

    Args:
        inputs_fixed  : list of QN-scaled integers, length = num_inputs
        weights_fixed : list of lists, shape (num_outputs, num_inputs),
                        QN-scaled integers
        biases_fixed  : list of QN-scaled integers, length = num_outputs
        apply_relu    : True for the hidden layer, False for the output layer

    Returns:
        outputs_fixed : list of QN-scaled integers, length = num_outputs,
                         saturated to the Q(qm, qn) representable range.

    Hardware note:
        Each product (weight * input) is scaled by 2^(2n), since BOTH
        operands are individually scaled by 2^n. We accumulate these
        full-precision products in a wide accumulator (in real hardware this
        would be a ~32/40-bit register, sized to avoid overflow while
        summing; Python integers are unbounded so we get this "for free"
        here). After accumulating, we shift right by n bits to rescale the
        sum back down to Q(qm, qn), then saturate to the final bit-width,
        then optionally apply ReLU -- in that exact order, matching a
        typical MAC -> shifter -> saturate -> ReLU hardware pipeline.
    """
    min_int, max_int = value_range(qm, qn)
    outputs_fixed = []

    for out_idx in range(len(weights_fixed)):
        acc = 0
        row = weights_fixed[out_idx]
        for in_idx, x in enumerate(inputs_fixed):
            acc += row[in_idx] * x                # scaled by 2^(2n)
        acc += biases_fixed[out_idx] << qn          # rescale bias to match product scale

        rescaled = acc >> qn                        # arithmetic right shift = divide by 2^n
        rescaled = max(min_int, min(max_int, rescaled))  # saturate to Q(qm, qn)

        if apply_relu:
            rescaled = max(0, rescaled)

        outputs_fixed.append(rescaled)

    return outputs_fixed

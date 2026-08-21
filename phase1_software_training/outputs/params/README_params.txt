Parameter export format
========================
INPUT_SIZE  = 64
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10

weights_hidden.txt : HIDDEN_SIZE x INPUT_SIZE values, row-major
                     (row h = weights of hidden neuron h)
bias_hidden.txt    : HIDDEN_SIZE values
weights_output.txt : OUTPUT_SIZE x HIDDEN_SIZE values, row-major
bias_output.txt    : OUTPUT_SIZE values
All values are plain decimal floating point (one per line).
Phase 2 will convert these into fixed-point Q(m,n) format.

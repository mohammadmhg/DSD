Fixed-point parameter export format
====================================
Format       : Q(8, 8)  -- 8 integer bits (incl. sign) + 8 fractional bits
Total width  : 16 bits, two's complement
Scale factor : 2^8 = 256

*_dec.txt : one signed decimal integer per line
*_hex.txt : one two's-complement hex string per line (paste into a VHDL ROM)

weights_hidden_* : HIDDEN_SIZE x INPUT_SIZE values, row-major
bias_hidden_*    : HIDDEN_SIZE values
weights_output_* : OUTPUT_SIZE x HIDDEN_SIZE values, row-major
bias_output_*    : OUTPUT_SIZE values

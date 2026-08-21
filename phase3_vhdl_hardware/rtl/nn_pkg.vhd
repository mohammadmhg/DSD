--------------------------------------------------------------------------------
-- nn_pkg.vhd
--
-- Shared package for the neural network accelerator: fixed-point format
-- constants, array types, and helper functions (saturate, rescale, relu).
--
-- The fixed-point format here MUST match Phase 2 (Python quantization):
--     Q(QM, QN) = Q(8, 8)  ->  16-bit signed, 8 integer bits (incl. sign),
--                              8 fractional bits.
-- If you change QM/QN here, you must re-run Phase 2 with the same QM/QN
-- and regenerate the ROMs (see scripts/generate_vhdl_roms.py).
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package nn_pkg is

    -- ---------------------------------------------------------------------
    -- Fixed-point format (must match Phase 2)
    -- ---------------------------------------------------------------------
    constant QM         : natural := 8;
    constant QN         : natural := 8;
    constant DATA_WIDTH : natural := QM + QN;  -- 16 bits

    -- Wide accumulator for the MAC unit. Each product of two DATA_WIDTH
    -- signed numbers needs 2*DATA_WIDTH bits; summing up to INPUT_SIZE (64)
    -- of them needs a few guard bits on top of that. 40 bits gives a
    -- comfortable safety margin without being wastefully large.
    constant ACC_WIDTH  : natural := 40;

    -- ---------------------------------------------------------------------
    -- Network dimensions (must match Phase 1/2 config.py)
    -- ---------------------------------------------------------------------
    constant INPUT_SIZE  : natural := 64;  -- 8x8 downsampled image
    constant HIDDEN_SIZE : natural := 16;
    constant OUTPUT_SIZE : natural := 10;  -- digits 0-9

    -- ---------------------------------------------------------------------
    -- Array types
    -- ---------------------------------------------------------------------
    type pixel_array_t  is array (0 to INPUT_SIZE-1)  of signed(DATA_WIDTH-1 downto 0);
    type hidden_array_t is array (0 to HIDDEN_SIZE-1) of signed(DATA_WIDTH-1 downto 0);
    type output_array_t is array (0 to OUTPUT_SIZE-1) of signed(DATA_WIDTH-1 downto 0);

    -- ---------------------------------------------------------------------
    -- Helper functions
    -- ---------------------------------------------------------------------

    -- Clamp (saturate) an arbitrary-width signed value down to
    -- `target_width` bits, instead of silently wrapping around on overflow.
    function saturate(value : signed; target_width : natural) return signed;

    -- Rescale a wide MAC accumulator (scaled by 2^(2*QN), because both the
    -- weight and the input were individually scaled by 2^QN) back down to
    -- the Q(QM, QN) format (scaled by 2^QN), with saturation. This mirrors
    -- Phase 2's fixed_point_layer() software model exactly.
    function rescale(acc : signed(ACC_WIDTH-1 downto 0)) return signed;

    -- ReLU activation: max(0, x)
    function relu(x : signed) return signed;

end package nn_pkg;


package body nn_pkg is

    function saturate(value : signed; target_width : natural) return signed is
        variable max_val : signed(value'range);
        variable min_val : signed(value'range);
        variable result  : signed(target_width-1 downto 0);
    begin
        max_val := to_signed(2**(target_width-1) - 1, value'length);
        min_val := to_signed(-(2**(target_width-1)), value'length);

        if value > max_val then
            result := to_signed(2**(target_width-1) - 1, target_width);
        elsif value < min_val then
            result := to_signed(-(2**(target_width-1)), target_width);
        else
            result := resize(value, target_width);
        end if;
        return result;
    end function;

    function rescale(acc : signed(ACC_WIDTH-1 downto 0)) return signed is
    begin
        -- shift_right on a SIGNED value performs an arithmetic shift
        -- (sign-extends), matching Python's ">>" on (possibly negative)
        -- integers used in Phase 2.
        return saturate(shift_right(acc, QN), DATA_WIDTH);
    end function;

    function relu(x : signed) return signed is
        variable zero : signed(x'range) := (others => '0');
    begin
        if x < 0 then
            return zero;
        else
            return x;
        end if;
    end function;

end package body nn_pkg;

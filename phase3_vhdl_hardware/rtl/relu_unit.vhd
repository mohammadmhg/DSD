--------------------------------------------------------------------------------
-- relu_unit.vhd
--
-- Hardware implementation of the ReLU activation function: f(x) = max(0, x).
-- Purely combinational -- a single comparator and multiplexer.
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity relu_unit is
    port (
        data_in  : in  signed(DATA_WIDTH-1 downto 0);
        data_out : out signed(DATA_WIDTH-1 downto 0)
    );
end entity relu_unit;

architecture rtl of relu_unit is
begin
    data_out <= data_in when data_in >= 0 else (others => '0');
end architecture rtl;

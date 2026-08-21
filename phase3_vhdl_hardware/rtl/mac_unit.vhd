--------------------------------------------------------------------------------
-- mac_unit.vhd
--
-- Multiply-Accumulate unit: the core building block of every neuron.
-- Computes acc = sum(a_in * b_in) over successive clock cycles, one
-- multiply-accumulate per cycle while `enable` is asserted.
--
-- clear_acc resets the accumulator to 0 (used at the start of each neuron's
-- computation, before its first MAC cycle).
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity mac_unit is
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        clear_acc : in  std_logic;                      -- synchronous clear
        enable    : in  std_logic;                      -- perform one MAC this cycle
        a_in      : in  signed(DATA_WIDTH-1 downto 0);  -- e.g. weight
        b_in      : in  signed(DATA_WIDTH-1 downto 0);  -- e.g. input activation
        acc_out   : out signed(ACC_WIDTH-1 downto 0)
    );
end entity mac_unit;

architecture rtl of mac_unit is
    signal acc_reg : signed(ACC_WIDTH-1 downto 0) := (others => '0');
begin

    process(clk, rst)
        variable product : signed(2*DATA_WIDTH-1 downto 0);
    begin
        if rst = '1' then
            acc_reg <= (others => '0');
        elsif rising_edge(clk) then
            if clear_acc = '1' then
                acc_reg <= (others => '0');
            elsif enable = '1' then
                product := a_in * b_in;  -- 2*DATA_WIDTH-bit signed product
                acc_reg <= acc_reg + resize(product, ACC_WIDTH);
            end if;
        end if;
    end process;

    acc_out <= acc_reg;

end architecture rtl;

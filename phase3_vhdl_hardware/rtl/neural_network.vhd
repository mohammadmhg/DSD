--------------------------------------------------------------------------------
-- neural_network.vhd
--
-- Top-level neural network accelerator.
--
-- Architecture: a single shared MAC unit is reused sequentially for every
-- neuron (one multiply-accumulate per clock cycle). This keeps hardware
-- resource usage small -- appropriate for a first, educational hardware
-- design -- at the cost of throughput (a full classification takes roughly
-- (INPUT_SIZE+3)*HIDDEN_SIZE + (HIDDEN_SIZE+3)*OUTPUT_SIZE + a few extra
-- cycles). A faster (but larger) design would use HIDDEN_SIZE parallel MAC
-- units; that trade-off is discussed in the final report.
--
-- Note on ReLU: the same max(0,x) computation performed by the standalone
-- relu_unit.vhd entity is applied here via the relu() function from nn_pkg
-- (functionally identical hardware -- a comparator + mux -- just inlined
-- for compactness inside the FSM's combinational logic). relu_unit.vhd is
-- kept as a separate, independently testable block for the report's block
-- diagram and for unit-level simulation.
--
-- FSM stages (see the project report for the full FSM diagram):
--   S_IDLE                      -- wait for `start`
--   S_H_INIT / S_H_MAC / S_H_BIAS / S_H_RESCALE / S_H_NEXT   -- hidden layer,
--       one pass per hidden neuron (16 passes)
--   S_O_INIT / S_O_MAC / S_O_BIAS / S_O_RESCALE / S_O_NEXT   -- output layer,
--       one pass per output neuron (10 passes)
--   S_ARGMAX_INIT / S_ARGMAX_STEP   -- find the winning class (largest logit)
--   S_DONE                       -- present the result for one cycle
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.nn_pkg.all;

entity neural_network is
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;                       -- synchronous, active-high
        start     : in  std_logic;                        -- pulse for 1 cycle to begin classification
        pixel_in  : in  pixel_array_t;                    -- the whole 8x8 (=64) quantized image at once
        done      : out std_logic;                         -- pulses high for 1 cycle when digit_out is valid
        digit_out : out unsigned(3 downto 0)                -- winning class, 0-9
    );
end entity neural_network;

architecture rtl of neural_network is

    type state_t is (
        S_IDLE,
        S_H_INIT, S_H_MAC, S_H_BIAS, S_H_RESCALE, S_H_NEXT,
        S_O_INIT, S_O_MAC, S_O_BIAS, S_O_RESCALE, S_O_NEXT,
        S_ARGMAX_INIT, S_ARGMAX_STEP,
        S_DONE
    );
    signal state : state_t := S_IDLE;

    signal input_reg  : pixel_array_t  := (others => (others => '0'));
    signal hidden_reg : hidden_array_t := (others => (others => '0'));
    signal output_reg : output_array_t := (others => (others => '0'));

    -- Reused counters: neuron_idx covers 0..HIDDEN_SIZE-1 during the hidden
    -- pass and 0..OUTPUT_SIZE-1 during the output pass; elem_idx covers
    -- 0..INPUT_SIZE-1 (hidden pass inputs) and 0..HIDDEN_SIZE-1 (output
    -- pass inputs). Both are sized for the larger of their two uses.
    signal neuron_idx : integer range 0 to HIDDEN_SIZE-1 := 0;
    signal elem_idx    : integer range 0 to INPUT_SIZE-1 := 0;

    -- MAC unit signals
    signal mac_clear : std_logic := '0';
    signal mac_en    : std_logic := '0';
    signal mac_a     : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    signal mac_b     : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    signal mac_acc   : signed(ACC_WIDTH-1 downto 0);

    -- ROM address / data signals
    signal wh_addr : integer range 0 to HIDDEN_SIZE*INPUT_SIZE-1 := 0;
    signal wh_data : signed(DATA_WIDTH-1 downto 0);
    signal bh_addr : integer range 0 to HIDDEN_SIZE-1 := 0;
    signal bh_data : signed(DATA_WIDTH-1 downto 0);
    signal wo_addr : integer range 0 to OUTPUT_SIZE*HIDDEN_SIZE-1 := 0;
    signal wo_data : signed(DATA_WIDTH-1 downto 0);
    signal bo_addr : integer range 0 to OUTPUT_SIZE-1 := 0;
    signal bo_data : signed(DATA_WIDTH-1 downto 0);

    -- Argmax tracking
    signal best_idx      : integer range 0 to OUTPUT_SIZE-1 := 0;
    signal best_val      : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    signal argmax_step_i : integer range 0 to OUTPUT_SIZE := 0;

begin

    -- Sub-block instances ---------------------------------------------------
    u_mac : entity work.mac_unit
        port map (
            clk => clk, rst => rst,
            clear_acc => mac_clear, enable => mac_en,
            a_in => mac_a, b_in => mac_b,
            acc_out => mac_acc
        );

    u_weights_hidden_rom : entity work.weights_hidden_rom
        port map (addr => wh_addr, data => wh_data);

    u_bias_hidden_rom : entity work.bias_hidden_rom
        port map (addr => bh_addr, data => bh_data);

    u_weights_output_rom : entity work.weights_output_rom
        port map (addr => wo_addr, data => wo_data);

    u_bias_output_rom : entity work.bias_output_rom
        port map (addr => bo_addr, data => bo_data);

    -- Main FSM ----------------------------------------------------------------
    process(clk, rst)
        variable rescaled : signed(DATA_WIDTH-1 downto 0);
    begin
        if rst = '1' then
            state         <= S_IDLE;
            done          <= '0';
            neuron_idx    <= 0;
            elem_idx      <= 0;
            mac_clear     <= '0';
            mac_en        <= '0';
            digit_out     <= (others => '0');
            argmax_step_i <= 0;

        elsif rising_edge(clk) then
            -- Defaults each cycle; explicitly overridden below where needed.
            mac_clear <= '0';
            mac_en    <= '0';
            done      <= '0';

            case state is

                ------------------------------------------------------------
                when S_IDLE =>
                    if start = '1' then
                        input_reg  <= pixel_in;
                        neuron_idx <= 0;
                        state      <= S_H_INIT;
                    end if;

                ------------------------------------------------------------
                -- Hidden layer: one full pass (64 MACs + bias + ReLU) per
                -- hidden neuron, repeated for all HIDDEN_SIZE neurons.
                ------------------------------------------------------------
                when S_H_INIT =>
                    mac_clear <= '1';
                    elem_idx  <= 0;
                    wh_addr   <= neuron_idx * INPUT_SIZE;  -- address of this neuron's 1st weight
                    state     <= S_H_MAC;

                when S_H_MAC =>
                    mac_a  <= wh_data;
                    mac_b  <= input_reg(elem_idx);
                    mac_en <= '1';
                    if elem_idx = INPUT_SIZE - 1 then
                        state <= S_H_BIAS;
                    else
                        elem_idx <= elem_idx + 1;
                        wh_addr  <= neuron_idx * INPUT_SIZE + elem_idx + 1;
                    end if;

                when S_H_BIAS =>
                    bh_addr <= neuron_idx;
                    state   <= S_H_RESCALE;

                when S_H_RESCALE =>
                    -- The bias is added at the same 2*QN scale as the MAC
                    -- products, so it is shifted left by QN first -- this
                    -- exactly mirrors Phase 2's fixed_point_layer().
                    rescaled := relu(rescale(mac_acc + shift_left(resize(bh_data, ACC_WIDTH), QN)));
                    hidden_reg(neuron_idx) <= rescaled;
                    state <= S_H_NEXT;

                when S_H_NEXT =>
                    if neuron_idx = HIDDEN_SIZE - 1 then
                        neuron_idx <= 0;
                        state <= S_O_INIT;
                    else
                        neuron_idx <= neuron_idx + 1;
                        state <= S_H_INIT;
                    end if;

                ------------------------------------------------------------
                -- Output layer: one full pass (16 MACs + bias, NO ReLU) per
                -- output neuron, repeated for all OUTPUT_SIZE neurons.
                ------------------------------------------------------------
                when S_O_INIT =>
                    mac_clear <= '1';
                    elem_idx  <= 0;
                    wo_addr   <= neuron_idx * HIDDEN_SIZE;
                    state     <= S_O_MAC;

                when S_O_MAC =>
                    mac_a  <= wo_data;
                    mac_b  <= hidden_reg(elem_idx);
                    mac_en <= '1';
                    if elem_idx = HIDDEN_SIZE - 1 then
                        state <= S_O_BIAS;
                    else
                        elem_idx <= elem_idx + 1;
                        wo_addr  <= neuron_idx * HIDDEN_SIZE + elem_idx + 1;
                    end if;

                when S_O_BIAS =>
                    bo_addr <= neuron_idx;
                    state   <= S_O_RESCALE;

                when S_O_RESCALE =>
                    -- No ReLU here: the output layer's raw scores (logits)
                    -- are compared directly in the argmax stage, exactly
                    -- like the Python reference model.
                    rescaled := rescale(mac_acc + shift_left(resize(bo_data, ACC_WIDTH), QN));
                    output_reg(neuron_idx) <= rescaled;
                    state <= S_O_NEXT;

                when S_O_NEXT =>
                    if neuron_idx = OUTPUT_SIZE - 1 then
                        state <= S_ARGMAX_INIT;
                    else
                        neuron_idx <= neuron_idx + 1;
                        state <= S_O_INIT;
                    end if;

                ------------------------------------------------------------
                -- Argmax: scan the 10 output logits and remember the
                -- largest one and its index -- that index IS the predicted
                -- digit.
                ------------------------------------------------------------
                when S_ARGMAX_INIT =>
                    best_idx      <= 0;
                    best_val      <= output_reg(0);
                    argmax_step_i <= 1;
                    state         <= S_ARGMAX_STEP;

                when S_ARGMAX_STEP =>
                    if argmax_step_i = OUTPUT_SIZE then
                        state <= S_DONE;
                    else
                        if output_reg(argmax_step_i) > best_val then
                            best_val <= output_reg(argmax_step_i);
                            best_idx <= argmax_step_i;
                        end if;
                        argmax_step_i <= argmax_step_i + 1;
                    end if;

                ------------------------------------------------------------
                when S_DONE =>
                    digit_out <= to_unsigned(best_idx, 4);
                    done      <= '1';
                    state     <= S_IDLE;

            end case;
        end if;
    end process;

end architecture rtl;

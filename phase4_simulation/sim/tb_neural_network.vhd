--------------------------------------------------------------------------------
-- tb_neural_network.vhd
--
-- Phase 4: comprehensive testbench.
--
-- Reads every exported test sample from Phase 2 (quantized test images, true
-- labels, and the Python fixed-point model's predictions), feeds each image
-- into the neural_network hardware entity, and reports:
--   - hardware vs. true-label accuracy
--   - hardware vs. Python fixed-point model agreement (this is the key
--     "does the hardware match the software model" check -- it should be
--     100% if Phase 3's design is correct, since both perform the exact
--     same integer arithmetic)
--   - classification latency, in clock cycles and in real time (assuming a
--     100 MHz clock, i.e. a 10 ns period -- change CLK_PERIOD below if your
--     target device uses a different frequency)
--
-- Run this via scripts/run_simulation.sh (see that script for why the file
-- paths below are relative the way they are).
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.nn_pkg.all;

entity tb_neural_network is
end entity tb_neural_network;

architecture sim of tb_neural_network is

    -- 100 MHz assumption; change this to match whatever clock your target
    -- FPGA board actually uses, and re-run to get updated latency numbers.
    constant CLK_PERIOD : time := 10 ns;

    signal clk       : std_logic := '0';
    signal rst       : std_logic := '1';
    signal start     : std_logic := '0';
    signal pixel_in  : pixel_array_t := (others => (others => '0'));
    signal done      : std_logic;
    signal digit_out : unsigned(3 downto 0);

    signal sim_finished : boolean := false;

    -- Paths are relative to the directory the simulation is RUN from.
    -- scripts/run_simulation.sh always runs GHDL from phase4_simulation/work/,
    -- so ../../ reaches the project root, from where these are correct.
    constant IMAGES_PATH : string := "../../phase2_quantization/outputs/test_samples_fixed/test_images_fixed_dec.txt";
    constant LABELS_PATH : string := "../../phase2_quantization/outputs/test_samples_fixed/test_labels.txt";
    constant PYPRED_PATH : string := "../../phase2_quantization/outputs/test_samples_fixed/hardware_predictions.txt";

begin

    -- Device under test ------------------------------------------------------
    dut : entity work.neural_network
        port map (
            clk => clk, rst => rst, start => start,
            pixel_in => pixel_in, done => done, digit_out => digit_out
        );

    clk <= not clk after CLK_PERIOD / 2 when not sim_finished else '0';

    -- Stimulus + checking ------------------------------------------------------
    stimulus : process
        file img_file    : text open read_mode is IMAGES_PATH;
        file label_file  : text open read_mode is LABELS_PATH;
        file pypred_file : text open read_mode is PYPRED_PATH;

        variable img_line, label_line, pypred_line : line;
        variable pixel_val    : integer;
        variable true_label   : integer;
        variable py_predicted : integer;

        variable sample_no       : integer := 0;
        variable correct_vs_true : integer := 0;
        variable matches_python  : integer := 0;

        variable t_start     : time;
        variable cycle_count : integer;
        variable total_cycles : integer := 0;
        variable min_cycles   : integer := 999999;
        variable max_cycles   : integer := 0;
    begin
        rst <= '1';
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait for CLK_PERIOD;

        report "===================================================================";
        report " Phase 4 Testbench -- neural_network hardware vs. software reference";
        report "===================================================================";

        while not endfile(img_file) loop
            readline(img_file, img_line);
            readline(label_file, label_line);
            readline(pypred_file, pypred_line);

            read(label_line, true_label);
            read(pypred_line, py_predicted);

            for i in 0 to INPUT_SIZE - 1 loop
                read(img_line, pixel_val);
                pixel_in(i) <= to_signed(pixel_val, DATA_WIDTH);
            end loop;

            wait for CLK_PERIOD;  -- let pixel_in settle before latching
            start   <= '1';
            t_start := now;
            wait for CLK_PERIOD;
            start <= '0';

            wait until done = '1';
            cycle_count := (now - t_start) / CLK_PERIOD;

            total_cycles := total_cycles + cycle_count;
            if cycle_count < min_cycles then min_cycles := cycle_count; end if;
            if cycle_count > max_cycles then max_cycles := cycle_count; end if;

            report "Sample " & integer'image(sample_no) &
                   " | true_label=" & integer'image(true_label) &
                   " | hw_predicted=" & integer'image(to_integer(digit_out)) &
                   " | python_predicted=" & integer'image(py_predicted) &
                   " | latency=" & integer'image(cycle_count) & " cycles";

            if to_integer(digit_out) = true_label then
                correct_vs_true := correct_vs_true + 1;
            end if;
            if to_integer(digit_out) = py_predicted then
                matches_python := matches_python + 1;
            end if;

            sample_no := sample_no + 1;
            wait for CLK_PERIOD * 3;  -- idle gap between samples, easier to read in the waveform
        end loop;

        report "===================================================================";
        report " SUMMARY (" & integer'image(sample_no) & " samples)";
        report "  Hardware accuracy vs. true labels          : " &
               integer'image(correct_vs_true) & " / " & integer'image(sample_no);
        report "  Hardware matches Python fixed-point model  : " &
               integer'image(matches_python) & " / " & integer'image(sample_no) &
               "  (should be 100% -- confirms Phase 3 matches Phase 2 exactly)";
        report "  Latency, clock cycles (min / avg / max)    : " &
               integer'image(min_cycles) & " / " &
               integer'image(total_cycles / sample_no) & " / " &
               integer'image(max_cycles);
        report "  Average latency @ 100 MHz (10 ns period)   : " &
               integer'image((total_cycles / sample_no) * 10) & " ns";
        report "===================================================================";

        sim_finished <= true;
        wait;
    end process;

end architecture sim;

@echo off
REM ------------------------------------------------------------------------------
REM run_simulation.bat
REM
REM Compiles the whole design (Phase 3 RTL + Phase 4 testbench) with GHDL and
REM runs the simulation, producing a waveform (.vcd) file you can open in
REM GTKWave (free, https://gtkwave.sourceforge.net/) or import into
REM ModelSim/Vivado.
REM
REM Usage:
REM     run_simulation.bat
REM ------------------------------------------------------------------------------

setlocal enabledelayedexpansion

REM Resolve directories relative to this script
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "PHASE4_DIR=%%~fI"
for %%I in ("%PHASE4_DIR%\..") do set "PROJECT_ROOT=%%~fI"

set "RTL_DIR=%PROJECT_ROOT%\phase3_vhdl_hardware\rtl"
set "WORK_DIR=%PHASE4_DIR%\work"
set "WAVE_DIR=%PHASE4_DIR%\waveforms"

if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"
if not exist "%WAVE_DIR%" mkdir "%WAVE_DIR%"

if not exist "%RTL_DIR%\roms\weights_hidden_rom.vhd" (
    echo ERROR: ROM files not found in %RTL_DIR%\roms\
    echo Run phase3_vhdl_hardware\scripts\generate_vhdl_roms.py first
    echo ^(after training in Phase 1 and quantizing in Phase 2^).
    exit /b 1
)

cd /d "%WORK_DIR%"

echo == Analyzing VHDL sources ==
ghdl -a --std=08 --work=work "%RTL_DIR%\nn_pkg.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\mac_unit.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\relu_unit.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\roms\weights_hidden_rom.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\roms\bias_hidden_rom.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\roms\weights_output_rom.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\roms\bias_output_rom.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%RTL_DIR%\neural_network.vhd"
if errorlevel 1 goto :error
ghdl -a --std=08 --work=work "%PHASE4_DIR%\sim\tb_neural_network.vhd"
if errorlevel 1 goto :error

echo.
echo == Elaborating ==
ghdl -e --std=08 --work=work tb_neural_network
if errorlevel 1 goto :error

echo.
echo == Running simulation ==
ghdl -r --std=08 --work=work tb_neural_network --vcd="%WAVE_DIR%\tb_neural_network.vcd"
if errorlevel 1 goto :error

echo.
echo Waveform saved to: %WAVE_DIR%\tb_neural_network.vcd
echo View it with GTKWave:  gtkwave "%WAVE_DIR%\tb_neural_network.vcd"
goto :end

:error
echo.
echo [ERROR] Simulation script failed at step above.
exit /b 1

:end
endlocal
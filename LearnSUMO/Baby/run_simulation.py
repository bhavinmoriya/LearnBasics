# filepath: c:\Users\moriyab\Downloads\Matvi\Simulation\Baby\run_simulation.py
import traci
import subprocess
import sys
import os

# SUMO_BINARY = "sumo"  # or "sumo-gui" for GUI
SUMO_BINARY = "sumo-gui"  # or "sumo-gui" for GUI

def run():
    sumo_cmd = [SUMO_BINARY, "-c", "config/simulation.sumocfg"]
    traci.start(sumo_cmd)
    step = 0
    while step < 100:
        traci.simulationStep()
        step += 1
    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    run()

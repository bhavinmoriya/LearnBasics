# Baby Blueprint: Minimal Mobility Simulation

## How to Run

1. Install SUMO and Python (with `traci`).
2. Place all files as shown above. In fact, cloning the REPO would actually put everything in the correct order.
3. Run the simulation:
   ```
   python run_simulation.py
   ```
4. View results in SUMO or process output as needed.

## Notes

- The included OSM and network files are minimal for demonstration.
- For real scenarios, use larger OSM/PBF files and generate networks with `netconvert`.
- Always use `netconvert` to get the `.net.xml` file. As the one generated otherwise will not go well with SUMO.

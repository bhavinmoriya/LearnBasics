# Basic flow
- Get `.osm.pbf`
- Convert OSM to net.xml: `netconvert --osm-files your-map.osm.pbf -o network.net.xml`    
- Generate routes: `python3 $SUMO_HOME/tools/randomTrips.py -n network.net.xml -o routes.rou.xml`    
- sumo.cfg:
	- ```xml
	<configuration>
	    <input>
	        <net-file value="network.net.xml"/>
	        <route-files value="routes.rou.xml"/>
	    </input>
	    <time>
	        <begin value="0"/>
	        <end value="1000"/>
	    </time>
	</configuration>
	```

# `randomTrips.py`
The file `randomTrips.py` is a utility script for [Eclipse SUMO (Simulation of Urban MObility)](https://eclipse.dev/sumo), a popular open-source traffic simulation package. Its main purpose is to **generate random trips** (vehicle or pedestrian routes) within a given road network, according to various customizable parameters and constraints.

### Core Purpose

- **Generates trips (start and end locations, optionally with intermediate waypoints) between random locations on a SUMO network.**
- These trips can be output as XML files for use in SUMO simulations.
- Can optionally run a route-finding tool (`duarouter` or `marouter`) to convert trips into actual routes on the network.

### Key Features

- **Supports vehicles and pedestrians:** Can generate trips for cars, buses, pedestrians, or custom vehicle classes.
- **Flexible trip generation:** Allows specification of minimum/maximum trip distances, randomization, edge weighting by length, speed, lanes, and more.
- **Flows and period control:** Can generate individual trips or flows (groups of trips with specified departure rates).
- **Edge selection and weighting:** Trip origins, destinations, and intermediate waypoints are chosen according to weightings (by length, lanes, speed, etc.) and you can restrict trips to certain edge types.
- **Validation:** Can optionally validate generated trips for connectivity using SUMO's routing tools.
- **Stops and intermodal trips:** Supports trips starting/ending at public transport stops or other infrastructure elements.
- **Additional customization:** Many command-line options for attributes, random factors, routes, output files, and more.

### How It Works

1. **Command-Line Parsing:** Parses a wide range of command-line options to configure trip generation.
2. **Network Loading:** Loads the SUMO network (from an XML file) and parses its edges and nodes.
3. **Edge Weight Calculation:** Calculates probabilities for each edge to be chosen as a source, destination, or intermediate, based on user-specified weighting schemes.
4. **Trip Sampling:** Randomly samples source and destination edges (and optionally intermediates) according to calculated weights, while enforcing distance and other constraints.
5. **Trip Writing:** Writes trips in SUMO's XML format to an output file. Optionally, runs a routing tool to generate route files.
6. **Validation (optional):** Uses SUMO’s routing tools to check if generated trips are valid and can be routed successfully.
7. **Advanced Features:** Supports person trips, flows, probability distributions for departures, and integration with additional infrastructure files.

### Example Use Cases

- **Generate a set of random vehicle trips for a traffic simulation scenario.**
- **Create random pedestrian flows for walkability studies.**
- **Benchmark routing algorithms with randomized trip sets.**
- **Simulate public transport usage by generating trips between stops.**

### Common Usage

Typically run via command line, e.g.:
```sh
python randomTrips.py -n network.net.xml -o trips.trips.xml --begin 0 --end 3600 --flows 100
```
This would generate 100 flows of random trips across the network for one hour.

---

**In summary:**  
`randomTrips.py` automates the creation of random trips and/or flows in a SUMO road network, supporting a wide range of customizations and constraints, to facilitate realistic traffic simulation scenarios.

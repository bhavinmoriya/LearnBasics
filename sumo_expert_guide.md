# Complete SUMO Mobility Simulation Expert Guide

## Table of Contents
1. [Introduction to SUMO](#introduction)
2. [Installation and Setup](#installation)
3. [Core Concepts and Architecture](#core-concepts)
4. [Building Your First Simulation](#first-simulation)
5. [Network Creation and Management](#network-creation)
6. [Traffic Demand Modeling](#traffic-demand)
7. [Advanced Vehicle and Driver Models](#vehicle-models)
8. [Traffic Light Control](#traffic-lights)
9. [Public Transport and Multimodal Simulation](#public-transport)
10. [Data Analysis and Visualization](#data-analysis)
11. [Performance Optimization](#performance)
12. [Integration with Other Tools](#integration)
13. [Real-World Applications](#applications)
14. [Troubleshooting and Best Practices](#troubleshooting)

## 1. Introduction to SUMO {#introduction}

SUMO (Simulation of Urban MObility) is an open-source, highly portable, microscopic and continuous multi-modal traffic simulation package designed to handle large road networks. Developed by the German Aerospace Center (DLR), SUMO is widely used in academia and industry for:

- Traffic management optimization
- Autonomous vehicle testing
- Smart city planning
- Environmental impact assessment
- Traffic flow analysis
- Public transport optimization

### Key Features
- **Microscopic simulation**: Individual vehicle behavior
- **Multi-modal**: Cars, trucks, buses, motorcycles, bicycles, pedestrians
- **Open source**: Free and customizable
- **Network import**: From OpenStreetMap, VISUM, Vissim, etc.
- **Real-time control**: Traffic lights, variable message signs
- **Extensive output**: Detectors, trajectories, emissions
- **Python integration**: TraCI (Traffic Control Interface)

## 2. Installation and Setup {#installation}

### Windows Installation
```bash
# Download from https://sumo.dlr.de/docs/Downloads.php
# Install the .msi package
# Add SUMO_HOME environment variable pointing to installation directory
```

### Linux Installation
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# From source
git clone https://github.com/eclipse/sumo.git
cd sumo
mkdir build/cmake-build && cd build/cmake-build
cmake ../..
make -j$(nproc)
```

### macOS Installation
```bash
# Using Homebrew
brew install sumo

# Or download from official website
```

### Environment Setup
```bash
# Add to .bashrc or .zshrc
export SUMO_HOME="/usr/share/sumo"  # Adjust path as needed
export PATH="$SUMO_HOME/bin:$PATH"
```

### Python Dependencies
```bash
pip install sumolib traci matplotlib numpy pandas
```

## 3. Core Concepts and Architecture {#core-concepts}

### Simulation Components

#### Network
- **Edges**: Road segments with lanes
- **Nodes**: Intersections and endpoints
- **Lanes**: Individual traffic lanes
- **Junctions**: Complex intersections
- **Connections**: Valid turning movements

#### Demand
- **Routes**: Predefined paths through network
- **Vehicles**: Individual traffic participants
- **Flows**: Continuous vehicle generation
- **Trips**: Origin-destination pairs

#### Infrastructure
- **Traffic Lights**: Signal control systems
- **Detectors**: Data collection points
- **Variable Message Signs**: Dynamic information
- **Parking Areas**: Vehicle storage

### File Types and Structure
```
simulation/
├── network.net.xml          # Road network
├── routes.rou.xml          # Vehicle routes and flows
├── additional.add.xml      # Infrastructure (detectors, etc.)
├── config.sumocfg          # Simulation configuration
├── tllogic.tll.xml        # Traffic light programs
└── output/
    ├── tripinfo.xml        # Trip statistics
    ├── detector.xml        # Detector measurements
    └── fcd.xml            # Floating car data
```

## 4. Building Your First Simulation {#first-simulation}

### Step 1: Create a Simple Network
```xml
<!-- simple.nod.xml - Define nodes -->
<nodes>
    <node id="1" x="0.0" y="0.0"/>
    <node id="2" x="100.0" y="0.0"/>
    <node id="3" x="200.0" y="0.0"/>
</nodes>
```

```xml
<!-- simple.edg.xml - Define edges -->
<edges>
    <edge id="1to2" from="1" to="2" priority="1" numLanes="1" speed="13.89"/>
    <edge id="2to3" from="2" to="3" priority="1" numLanes="1" speed="13.89"/>
</edges>
```

### Step 2: Generate Network
```bash
netconvert --node-files=simple.nod.xml --edge-files=simple.edg.xml --output-file=simple.net.xml
```

### Step 3: Create Traffic Demand
```xml
<!-- simple.rou.xml -->
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="55"/>
    
    <route id="route1" edges="1to2 2to3"/>
    
    <vehicle id="veh1" type="car" route="route1" depart="0"/>
    <vehicle id="veh2" type="car" route="route1" depart="5"/>
    <vehicle id="veh3" type="car" route="route1" depart="10"/>
</routes>
```

### Step 4: Configuration File
```xml
<!-- simple.sumocfg -->
<configuration>
    <input>
        <net-file value="simple.net.xml"/>
        <route-files value="simple.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="100"/>
        <step-length value="0.1"/>
    </time>
    
    <output>
        <tripinfo-output value="tripinfo.xml"/>
    </output>
</configuration>
```

### Step 5: Run Simulation
```bash
sumo -c simple.sumocfg
# Or with GUI
sumo-gui -c simple.sumocfg
```

## 5. Network Creation and Management {#network-creation}

### Importing from OpenStreetMap
```bash
# Download area from OpenStreetMap
# Using osmWebWizard.py
cd $SUMO_HOME/tools
python osmWebWizard.py

# Or manually
wget "https://api.openstreetmap.org/api/0.6/map?bbox=left,bottom,right,top" -O map.osm
netconvert --osm-files map.osm -o network.net.xml
```

### Network Editing with NETEDIT
```bash
# Launch network editor
netedit network.net.xml
```

Key NETEDIT operations:
- **Create Mode**: Add nodes and edges
- **Delete Mode**: Remove elements
- **Inspect Mode**: View/edit properties
- **Select Mode**: Multi-element operations
- **Move Mode**: Reposition elements

### Advanced Network Features

#### Lane-specific Attributes
```xml
<edge id="highway" from="A" to="B">
    <lane id="highway_0" index="0" speed="25" allow="bus taxi"/>
    <lane id="highway_1" index="1" speed="35" allow="passenger"/>
    <lane id="highway_2" index="2" speed="35" allow="passenger delivery"/>
</edge>
```

#### Connection Control
```xml
<connection from="edge1" to="edge2" fromLane="0" toLane="1" 
           via="junction_0" tl="tls1" linkIndex="5"/>
```

#### Roundabouts
```bash
netconvert --roundabouts.guess=true --osm-files=input.osm -o output.net.xml
```

## 6. Traffic Demand Modeling {#traffic-demand}

### Vehicle Types and Classes
```xml
<vType id="passenger" vClass="passenger" 
       length="4.5" width="2.0" height="1.5"
       accel="2.6" decel="4.5" sigma="0.5"
       maxSpeed="55" minGap="2.5" tau="1.0"/>

<vType id="truck" vClass="truck"
       length="12.0" width="2.5" height="3.5"
       accel="1.0" decel="4.0" sigma="0.5"
       maxSpeed="25" minGap="3.0" tau="1.5"/>

<vType id="bus" vClass="bus"
       length="12.0" width="2.5" height="3.2"
       accel="1.2" decel="4.0" sigma="0.5"
       maxSpeed="20" minGap="3.0" tau="1.0"
       personCapacity="85" boardingDuration="0.5"/>
```

### Flow Generation
```xml
<!-- Constant flow -->
<flow id="flow1" type="passenger" route="route1" 
      begin="0" end="3600" vehsPerHour="1800"/>

<!-- Poisson distributed -->
<flow id="flow2" type="passenger" route="route2"
      begin="0" end="7200" period="2.0"/>

<!-- Variable flow with time intervals -->
<flow id="flow3" type="passenger" route="route3" begin="0" end="10800">
    <param key="vehsPerHour.0" value="600"/>      <!-- 0-1800s -->
    <param key="vehsPerHour.1800" value="1200"/>  <!-- 1800-3600s -->
    <param key="vehsPerHour.3600" value="1800"/>  <!-- 3600-5400s -->
</flow>
```

### Origin-Destination Matrices
```python
# Generate OD matrix with Python
import sumolib

# Create OD matrix
od_matrix = {
    ('origin1', 'dest1'): 150,
    ('origin1', 'dest2'): 200,
    ('origin2', 'dest1'): 100,
    ('origin2', 'dest2'): 250
}

# Convert to SUMO format
sumolib.output.convert_od_to_trips(od_matrix, "trips.xml")
```

### Route Generation Tools
```bash
# Generate routes using DUAROUTER
duarouter -n network.net.xml -t trips.xml -o routes.xml

# With turn-around prohibition
duarouter -n network.net.xml -t trips.xml -o routes.xml --ignore-errors --repair

# Dynamic User Assignment (DUA)
python $SUMO_HOME/tools/assign/duaIterate.py -n network.net.xml -t trips.xml -l 10
```

## 7. Advanced Vehicle and Driver Models {#vehicle-models}

### Car-Following Models

#### Krauss Model (Default)
```xml
<vType id="krauss" carFollowModel="Krauss"
       accel="2.6" decel="4.5" sigma="0.5" tau="1.0"/>
```

#### IDM (Intelligent Driver Model)
```xml
<vType id="idm" carFollowModel="IDM"
       accel="2.6" decel="4.5" tau="1.6"
       delta="4" stepping="0.25"/>
```

#### CACC (Cooperative Adaptive Cruise Control)
```xml
<vType id="cacc" carFollowModel="CACC"
       accel="2.6" decel="4.5" tau="1.0"
       sc="4" cc1="1.3" cc2="7.0"/>
```

### Lane-Changing Models

#### LC2013 (Default)
```xml
<vType id="aggressive" lcStrategic="1.0" lcCooperative="1.0"
       lcSpeedGain="1.0" lcKeepRight="1.0"/>
```

#### SL2015 (Sublane Model)
```xml
<vType id="sublane" carFollowModel="Krauss" laneChangeModel="SL2015"
       lcSublane="1.0" lcPushy="0" lcAssertive="1"/>
```

### Junction Models
```xml
<vType id="cautious" jmCrossingGap="10" jmIgnoreKeepClearTime="10"
       jmDriveAfterRedTime="0" jmDriveRedSpeed="5.56"/>
```

### Electric and Autonomous Vehicles
```xml
<!-- Electric vehicle -->
<vType id="electric" vClass="evehicle"
       maximumBatteryCapacity="35000" actualBatteryCapacity="35000"
       energyConsumption="150"/>

<!-- Autonomous vehicle -->
<vType id="av" carFollowModel="CACC" 
       minGap="1.0" tau="0.5" sigma="0"/>
```

## 8. Traffic Light Control {#traffic-lights}

### Static Traffic Light Programs
```xml
<tlLogic id="tls1" type="static" programID="0" offset="0">
    <phase duration="31" state="GGGrrrrrr"/>
    <phase duration="6"  state="yyyrrrrr"/>
    <phase duration="31" state="rrrGGGrrr"/>
    <phase duration="6"  state="rrryyyrrr"/>
    <phase duration="31" state="rrrrrrGGG"/>
    <phase duration="6"  state="rrrrrryy"/>
</tlLogic>
```

### Actuated Traffic Lights
```xml
<tlLogic id="tls1" type="actuated" programID="0">
    <param key="max-gap" value="3.0"/>
    <param key="detector-gap" value="2.0"/>
    <param key="passing-time" value="1.9"/>
    
    <phase duration="5" state="GGGrrrr" minDur="5" maxDur="50"/>
    <phase duration="3" state="yyyrrr" minDur="3" maxDur="3"/>
    <phase duration="5" state="rrrGGG" minDur="5" maxDur="50"/>
    <phase duration="3" state="rrryyy" minDur="3" maxDur="3"/>
</tlLogic>
```

### Induction Loop Detectors
```xml
<additionalFile>
    <inductionLoop id="det1" lane="edge1_0" pos="100" freq="60" file="detector.xml"/>
    <inductionLoop id="det2" lane="edge2_0" pos="50" freq="60" file="detector.xml"/>
</additionalFile>
```

### SCATS and SCOOT Integration
```python
# TraCI control example
import traci

# Connect to SUMO
traci.start(["sumo", "-c", "config.sumocfg"])

# Control traffic light
while traci.simulation.getMinExpectedNumber() > 0:
    # Get detector data
    vehicle_count = traci.inductionloop.getLastStepVehicleNumber("det1")
    
    # Adaptive control logic
    if vehicle_count > 10:
        traci.trafficlight.setPhase("tls1", 0)  # Green phase
    
    traci.simulationStep()

traci.close()
```

## 9. Public Transport and Multimodal Simulation {#public-transport}

### Bus Routes and Stops
```xml
<!-- Bus stops -->
<busStop id="stop1" lane="edge1_0" startPos="50" endPos="70" 
         lines="bus1 bus2" personCapacity="20"/>

<!-- Bus routes -->
<route id="busRoute1" edges="edge1 edge2 edge3">
    <stop busStop="stop1" duration="20" until="60"/>
    <stop busStop="stop2" duration="15" until="120"/>
</route>

<!-- Bus vehicles -->
<vehicle id="bus1" type="bus" route="busRoute1" depart="0" line="1"/>
```

### Trains and Railways
```xml
<!-- Train type -->
<vType id="train" vClass="rail" length="100" maxSpeed="40"
       accel="1.0" decel="1.0" sigma="0"/>

<!-- Railway edge -->
<edge id="railway" from="station1" to="station2" 
      allow="rail" speed="40" priority="1"/>

<!-- Train stop -->
<trainStop id="platform1" lane="railway_0" startPos="100" endPos="200"
           lines="train1"/>
```

### Pedestrian Simulation
```xml
<!-- Pedestrian crossings -->
<crossing id="crossing1" edges="edge1 edge2" priority="1" width="4"/>

<!-- Pedestrian areas -->
<pedestrianCrossing id="zebra1" edges="edge1" priority="1" width="4"/>

<!-- Pedestrians -->
<person id="ped1" depart="0">
    <walk from="edge1" to="edge2" duration="60"/>
</person>
```

### Intermodal Routing
```python
# Generate intermodal trips
import os
intermodal_script = os.path.join(os.environ['SUMO_HOME'], 'tools', 'route', 'intermodal.py')

os.system(f"python {intermodal_script} -n network.net.xml -e 7200 "
          "--trip-files trips.xml -o intermodal.rou.xml "
          "--persontrips --bike --car --public")
```

## 10. Data Analysis and Visualization {#data-analysis}

### Output Files and Analysis

#### Trip Information Analysis
```python
import xml.etree.ElementTree as ET
import pandas as pd

# Parse tripinfo.xml
tree = ET.parse('tripinfo.xml')
root = tree.getroot()

trips = []
for trip in root.findall('tripinfo'):
    trips.append({
        'id': trip.get('id'),
        'depart': float(trip.get('depart')),
        'arrival': float(trip.get('arrival')),
        'duration': float(trip.get('duration')),
        'routeLength': float(trip.get('routeLength')),
        'waitingTime': float(trip.get('waitingTime')),
        'timeLoss': float(trip.get('timeLoss'))
    })

df = pd.DataFrame(trips)
print(f"Average travel time: {df['duration'].mean():.2f} seconds")
print(f"Average waiting time: {df['waitingTime'].mean():.2f} seconds")
```

#### Detector Data Analysis
```python
# Analyze detector measurements
detector_data = []
for detector in root.findall('detector'):
    detector_data.append({
        'id': detector.get('id'),
        'nVehContrib': int(detector.get('nVehContrib')),
        'flow': float(detector.get('flow')),
        'occupancy': float(detector.get('occupancy')),
        'speed': float(detector.get('speed'))
    })

detector_df = pd.DataFrame(detector_data)
```

### Visualization with matplotlib
```python
import matplotlib.pyplot as plt

# Travel time distribution
plt.figure(figsize=(10, 6))
plt.hist(df['duration'], bins=30, alpha=0.7)
plt.xlabel('Travel Time (seconds)')
plt.ylabel('Number of Vehicles')
plt.title('Travel Time Distribution')
plt.show()

# Flow over time
plt.figure(figsize=(12, 6))
plt.plot(detector_df['id'], detector_df['flow'], 'o-')
plt.xlabel('Detector ID')
plt.ylabel('Flow (veh/h)')
plt.title('Traffic Flow by Detector')
plt.xticks(rotation=45)
plt.show()
```

### Real-time Visualization with TraCI
```python
import traci
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Real-time plotting setup
fig, ax = plt.subplots()
vehicles_x, vehicles_y = [], []

def update_plot(frame):
    vehicles_x.clear()
    vehicles_y.clear()
    
    for veh_id in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(veh_id)
        vehicles_x.append(x)
        vehicles_y.append(y)
    
    ax.clear()
    ax.scatter(vehicles_x, vehicles_y, c='red', s=20)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    
    traci.simulationStep()

# Start animation
traci.start(["sumo", "-c", "config.sumocfg"])
anim = FuncAnimation(fig, update_plot, interval=100)
plt.show()
traci.close()
```

## 11. Performance Optimization {#performance}

### Simulation Performance Tips

#### Network Optimization
- Remove unnecessary detail from imported networks
- Use appropriate lane numbers
- Simplify junction geometry
- Remove unused roads

```bash
# Network simplification
netconvert --osm-files input.osm -o output.net.xml \
           --remove-edges.isolated \
           --junctions.join \
           --tls.guess \
           --ramps.guess
```

#### Demand Optimization
- Use appropriate time intervals
- Calibrate vehicle types
- Optimize route distribution
- Use flow instead of individual vehicles when possible

```xml
<!-- Efficient flow definition -->
<flow id="highway_flow" type="car" route="highway" 
      begin="0" end="7200" vehsPerHour="2000" 
      departSpeed="max" departLane="best"/>
```

#### Simulation Parameters
```xml
<configuration>
    <processing>
        <step-length value="1.0"/>          <!-- Increase for speed -->
        <lateral-resolution value="1.28"/>   <!-- Decrease for speed -->
        <threads value="4"/>                 <!-- Use multiple cores -->
        <lanechange.duration value="3"/>     <!-- Faster lane changes -->
    </processing>
</configuration>
```

### Memory Management
```python
# Efficient TraCI usage
import traci

# Use vehicle subscriptions for better performance
traci.start(["sumo", "-c", "config.sumocfg"])

# Subscribe to specific variables only
VAR_POSITION = 0x42
VAR_SPEED = 0x40
for veh_id in traci.vehicle.getIDList():
    traci.vehicle.subscribe(veh_id, [VAR_POSITION, VAR_SPEED])

# Process subscription results
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    subscription_results = traci.vehicle.getSubscriptionResults()
    
    for veh_id, data in subscription_results.items():
        position = data[VAR_POSITION]
        speed = data[VAR_SPEED]
        # Process data efficiently

traci.close()
```

### Parallel Simulation
```bash
# Run multiple scenarios in parallel
sumo --configuration-file scenario1.sumocfg &
sumo --configuration-file scenario2.sumocfg &
sumo --configuration-file scenario3.sumocfg &
wait
```

## 12. Integration with Other Tools {#integration}

### SUMO with Python (Advanced TraCI)
```python
import traci
import numpy as np

class SUMOController:
    def __init__(self, config_file):
        self.config_file = config_file
        self.vehicle_data = {}
    
    def start_simulation(self):
        traci.start(["sumo", "-c", self.config_file])
    
    def run_step(self):
        # Collect vehicle data
        for veh_id in traci.vehicle.getIDList():
            self.vehicle_data[veh_id] = {
                'position': traci.vehicle.getPosition(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'lane': traci.vehicle.getLaneID(veh_id),
                'waiting_time': traci.vehicle.getWaitingTime(veh_id)
            }
        
        # Apply control logic
        self.adaptive_speed_control()
        self.dynamic_routing()
        
        traci.simulationStep()
    
    def adaptive_speed_control(self):
        for veh_id in self.vehicle_data:
            current_speed = self.vehicle_data[veh_id]['speed']
            if current_speed < 5:  # Traffic jam condition
                # Find alternative route
                current_route = traci.vehicle.getRoute(veh_id)
                # Implement rerouting logic
    
    def dynamic_routing(self):
        # Implement dynamic traffic assignment
        pass
    
    def close_simulation(self):
        traci.close()

# Usage
controller = SUMOController("config.sumocfg")
controller.start_simulation()

while traci.simulation.getMinExpectedNumber() > 0:
    controller.run_step()

controller.close_simulation()
```

### Integration with MATLAB
```matlab
% MATLAB-SUMO integration
function sumo_matlab_interface()
    % Start SUMO with TraCI
    system('sumo --remote-port 8813 -c config.sumocfg &');
    pause(2); % Wait for SUMO to start
    
    % Connect to SUMO (requires TraCI4Matlab)
    traci_interface = TraCI('localhost', 8813);
    
    % Simulation loop
    while traci_interface.simulation.getMinExpectedNumber() > 0
        % Get vehicle data
        vehicle_ids = traci_interface.vehicle.getIDList();
        
        for i = 1:length(vehicle_ids)
            veh_id = vehicle_ids{i};
            position = traci_interface.vehicle.getPosition(veh_id);
            speed = traci_interface.vehicle.getSpeed(veh_id);
            
            % Apply MATLAB control algorithms
            new_speed = control_algorithm(speed, position);
            traci_interface.vehicle.setSpeed(veh_id, new_speed);
        end
        
        traci_interface.simulationStep();
    end
    
    traci_interface.close();
end

function new_speed = control_algorithm(current_speed, position)
    % Implement your control logic here
    new_speed = min(current_speed * 1.1, 15); % Simple acceleration
end
```

### SUMO with ROS (Robot Operating System)
```python
#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import traci

class SUMORosInterface:
    def __init__(self):
        rospy.init_node('sumo_ros_interface')
        self.pub = rospy.Publisher('/vehicle/odom', Odometry, queue_size=10)
        self.sub = rospy.Subscriber('/vehicle/cmd_vel', Twist, self.cmd_callback)
        
        # Start SUMO
        traci.start(["sumo", "-c", "config.sumocfg"])
        
        # Main loop
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown() and traci.simulation.getMinExpectedNumber() > 0:
            self.publish_vehicle_state()
            traci.simulationStep()
            rate.sleep()
        
        traci.close()
    
    def publish_vehicle_state(self):
        for veh_id in traci.vehicle.getIDList():
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"
            
            x, y = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            angle = traci.vehicle.getAngle(veh_id)
            
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.twist.twist.linear.x = speed
            
            self.pub.publish(odom)
    
    def cmd_callback(self, msg):
        # Apply velocity commands to SUMO vehicles
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(veh_id, msg.linear.x)

if __name__ == '__main__':
    SUMORosInterface()
```

## 13. Real-World Applications {#applications}

### Traffic Signal Optimization
```python
# Genetic Algorithm for Traffic Light Optimization
import random
import traci
import numpy as np

class TrafficLightGA:
    def __init__(self, tl_id, phases):
        self.tl_id = tl_id
        self.phases = phases
        self.population_size = 20
        self.generations = 50
    
    def create_individual(self):
        # Create random phase durations
        return [random.randint(10, 60) for _ in self.phases]
    
    def evaluate_fitness(self, individual):
        # Set traffic light phases
        program = []
        for i, duration in enumerate(individual):
            phase_def = traci.trafficlight.getPhase(self.tl_id, i)
            phase_def.duration = duration
            program.append(phase_def)
        
        # Run simulation and measure performance
        traci.trafficlight.setProgramLogic(self.tl_id, program)
        
        total_waiting_time = 0
        steps = 0
        while steps < 3600:  # 1 hour simulation
            total_waiting_time += sum(traci.vehicle.getWaitingTime(v) 
                                    for v in traci.vehicle.getIDList())
            traci.simulationStep()
            steps += 1
        
        return 1.0 / (total_waiting_time + 1)  # Fitness is inverse of waiting time
    
    def optimize(self):
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            
            # Selection, crossover, mutation
            new_population = self.genetic_operations(population, fitness_scores)
            population = new_population
        
        # Return best individual
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]
```

### Connected and Autonomous Vehicles
```python
# V2X Communication Simulation
class V2XController:
    def __init__(self):
        self.vehicle_info = {}
        self.communication_range = 300  # meters
    
    def update_vehicle_info(self):
        for veh_id in traci.vehicle.getIDList():
            self.vehicle_info[veh_id] = {
                'position': traci.vehicle.getPosition(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id),
                'lane': traci.vehicle.getLaneID(veh_id),
                'route': traci.vehicle.getRoute(veh_id)
            }
    
    def vehicle_to_vehicle_communication(self, veh_id):
        neighbors = []
        veh_pos = self.vehicle_info[veh_id]['position']
        
        for other_id, other_info in self.vehicle_info.items():
            if other_id == veh_id:
                continue
            
            distance = np.sqrt((veh_pos[0] - other_info['position'][0])**2 + 
                              (veh_pos[1] - other_info['position'][1])**2)
            
            if distance <= self.communication_range:
                neighbors.append({
                    'id': other_id,
                    'distance': distance,
                    'info': other_info
                })
        
        return neighbors
    
    def cooperative_adaptive_cruise_control(self, veh_id):
        neighbors = self.vehicle_to_vehicle_communication(veh_id)
        current_speed = self.vehicle_info[veh_id]['speed']
        
        # Find vehicle directly ahead
        ahead_vehicle = None
        min_distance = float('inf')
        
        for neighbor in neighbors:
            if neighbor['info']['lane'] == self.vehicle_info[veh_id]['lane']:
                if neighbor['distance'] < min_distance:
                    min_distance = neighbor['distance']
                    ahead_vehicle = neighbor
        
        if ahead_vehicle:
            # CACC algorithm
            desired_gap = 2.0  # seconds
            leader_speed = ahead_vehicle['info']['speed']
            gap = min_distance
            desired_distance = current_speed * desired_gap
            
            # Simple CACC control law
            speed_error = leader_speed - current_speed
            gap_error = gap - desired_distance
            
            new_speed = current_speed + 0.1 * speed_error + 0.05 * gap_error
            new_speed = max(0, min(new_speed, 30))  # Speed limits
            
            traci.vehicle.setSpeed(veh_id, new_speed)
    
    def intersection_management(self, intersection_id):
        # Get vehicles approaching intersection
        approaching_vehicles = []
        
        for veh_id in traci.vehicle.getIDList():
            next_tls = traci.vehicle.getNextTLS(veh_id)
            if next_tls and next_tls[0][0] == intersection_id:
                distance_to_tls = next_tls[0][2]
                if distance_to_tls < 200:  # Within 200m
                    approaching_vehicles.append({
                        'id': veh_id,
                        'distance': distance_to_tls,
                        'speed': traci.vehicle.getSpeed(veh_id),
                        'lane': traci.vehicle.getLaneID(veh_id)
                    })
        
        # Optimize intersection timing
        self.optimize_intersection_timing(intersection_id, approaching_vehicles)
    
    def optimize_intersection_timing(self, intersection_id, vehicles):
        # Simple optimization: give more time to lanes with more vehicles
        lane_counts = {}
        for veh in vehicles:
            lane = veh['lane']
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
        
        # Adjust phase durations based on demand
        current_program = traci.trafficlight.getAllProgramLogics(intersection_id)[0]
        for i, phase in enumerate(current_program.phases):
            # Simplified logic - in practice, this would be more sophisticated
            if lane_counts:
                max_count = max(lane_counts.values())
                phase.duration = max(15, min(60, 20 + max_count * 2))

# Usage
v2x = V2XController()
while traci.simulation.getMinExpectedNumber() > 0:
    v2x.update_vehicle_info()
    
    for veh_id in traci.vehicle.getIDList():
        v2x.cooperative_adaptive_cruise_control(veh_id)
    
    traci.simulationStep()
```

### Environmental Impact Assessment
```python
# Emissions and Fuel Consumption Analysis
class EnvironmentalAnalyzer:
    def __init__(self):
        self.emissions_data = {}
        self.fuel_data = {}
    
    def collect_emissions(self):
        for veh_id in traci.vehicle.getIDList():
            # Get emission values
            co2 = traci.vehicle.getCO2Emission(veh_id)
            co = traci.vehicle.getCOEmission(veh_id)
            hc = traci.vehicle.getHCEmission(veh_id)
            pmx = traci.vehicle.getPMxEmission(veh_id)
            nox = traci.vehicle.getNOxEmission(veh_id)
            fuel = traci.vehicle.getFuelConsumption(veh_id)
            
            if veh_id not in self.emissions_data:
                self.emissions_data[veh_id] = {
                    'CO2': [], 'CO': [], 'HC': [], 'PMx': [], 'NOx': []
                }
                self.fuel_data[veh_id] = []
            
            self.emissions_data[veh_id]['CO2'].append(co2)
            self.emissions_data[veh_id]['CO'].append(co)
            self.emissions_data[veh_id]['HC'].append(hc)
            self.emissions_data[veh_id]['PMx'].append(pmx)
            self.emissions_data[veh_id]['NOx'].append(nox)
            self.fuel_data[veh_id].append(fuel)
    
    def analyze_environmental_impact(self):
        total_emissions = {'CO2': 0, 'CO': 0, 'HC': 0, 'PMx': 0, 'NOx': 0}
        total_fuel = 0
        
        for veh_id in self.emissions_data:
            for pollutant in total_emissions:
                total_emissions[pollutant] += sum(self.emissions_data[veh_id][pollutant])
            total_fuel += sum(self.fuel_data[veh_id])
        
        return total_emissions, total_fuel
    
    def compare_scenarios(self, baseline_emissions, scenario_emissions):
        improvement = {}
        for pollutant in baseline_emissions:
            if baseline_emissions[pollutant] > 0:
                reduction = (baseline_emissions[pollutant] - scenario_emissions[pollutant]) / baseline_emissions[pollutant]
                improvement[pollutant] = reduction * 100  # Percentage improvement
        
        return improvement

# Electric Vehicle Integration
class EVChargingSimulation:
    def __init__(self):
        self.charging_stations = {}
        self.ev_batteries = {}
    
    def setup_charging_infrastructure(self, stations_config):
        for station_id, config in stations_config.items():
            self.charging_stations[station_id] = {
                'position': config['position'],
                'capacity': config['capacity'],
                'power': config['power'],  # kW
                'occupied': 0,
                'queue': []
            }
    
    def manage_ev_charging(self):
        for veh_id in traci.vehicle.getIDList():
            veh_type = traci.vehicle.getTypeID(veh_id)
            
            if 'electric' in veh_type or 'ev' in veh_type:
                battery_level = traci.vehicle.getParameter(veh_id, "actualBatteryCapacity")
                max_battery = traci.vehicle.getParameter(veh_id, "maximumBatteryCapacity")
                
                battery_percentage = float(battery_level) / float(max_battery)
                
                # Check if vehicle needs charging
                if battery_percentage < 0.2:  # Less than 20%
                    nearest_station = self.find_nearest_charging_station(veh_id)
                    if nearest_station:
                        self.route_to_charging_station(veh_id, nearest_station)
    
    def find_nearest_charging_station(self, veh_id):
        veh_pos = traci.vehicle.getPosition(veh_id)
        min_distance = float('inf')
        nearest_station = None
        
        for station_id, station in self.charging_stations.items():
            distance = np.sqrt((veh_pos[0] - station['position'][0])**2 + 
                              (veh_pos[1] - station['position'][1])**2)
            
            if distance < min_distance and station['occupied'] < station['capacity']:
                min_distance = distance
                nearest_station = station_id
        
        return nearest_station
    
    def route_to_charging_station(self, veh_id, station_id):
        station_pos = self.charging_stations[station_id]['position']
        # Find nearest edge to charging station
        # This is simplified - in practice, you'd use proper routing
        current_route = traci.vehicle.getRoute(veh_id)
        # Implement routing to charging station
```

### Smart City Integration
```python
# Smart Traffic Management System
class SmartCityManager:
    def __init__(self):
        self.traffic_data = {}
        self.incidents = {}
        self.vms_messages = {}  # Variable Message Signs
        self.adaptive_signals = {}
    
    def collect_city_wide_data(self):
        # Collect data from all detectors
        detector_data = {}
        for det_id in traci.inductionloop.getIDList():
            detector_data[det_id] = {
                'count': traci.inductionloop.getLastStepVehicleNumber(det_id),
                'occupancy': traci.inductionloop.getLastStepOccupancy(det_id),
                'mean_speed': traci.inductionloop.getLastStepMeanSpeed(det_id)
            }
        
        # Collect traffic light states
        tl_data = {}
        for tl_id in traci.trafficlight.getIDList():
            tl_data[tl_id] = {
                'phase': traci.trafficlight.getPhase(tl_id),
                'next_switch': traci.trafficlight.getNextSwitch(tl_id)
            }
        
        return detector_data, tl_data
    
    def incident_detection(self, detector_data):
        # Simple incident detection based on occupancy and speed
        for det_id, data in detector_data.items():
            if data['occupancy'] > 80 and data['mean_speed'] < 5:
                if det_id not in self.incidents:
                    self.incidents[det_id] = {
                        'start_time': traci.simulation.getTime(),
                        'severity': 'high' if data['occupancy'] > 90 else 'medium'
                    }
                    self.respond_to_incident(det_id)
            else:
                # Clear incident if conditions improve
                if det_id in self.incidents:
                    del self.incidents[det_id]
                    self.clear_incident_response(det_id)
    
    def respond_to_incident(self, detector_id):
        # Update VMS messages
        message = "INCIDENT AHEAD - USE ALT ROUTE"
        # In practice, you'd identify upstream VMS signs
        
        # Adjust traffic light timings
        # Increase green time for alternative routes
        
        # Implement dynamic routing
        self.implement_dynamic_routing(detector_id)
    
    def implement_dynamic_routing(self, incident_location):
        # Reroute vehicles away from incident
        for veh_id in traci.vehicle.getIDList():
            current_route = traci.vehicle.getRoute(veh_id)
            route_index = traci.vehicle.getRouteIndex(veh_id)
            
            # Check if vehicle's route passes through incident area
            # This is simplified - real implementation would be more complex
            if self.route_passes_incident(current_route, incident_location):
                # Calculate alternative route
                origin = current_route[route_index]
                destination = current_route[-1]
                
                # Use DUAROUTER to find alternative
                alt_route = self.find_alternative_route(origin, destination, incident_location)
                if alt_route:
                    traci.vehicle.setRoute(veh_id, alt_route)
    
    def adaptive_signal_control(self, detector_data, tl_data):
        # Implement SCATS-like adaptive control
        for tl_id in tl_data:
            # Get detectors associated with this traffic light
            associated_detectors = self.get_tl_detectors(tl_id)
            
            # Calculate demand for each approach
            approach_demand = {}
            for det_id in associated_detectors:
                if det_id in detector_data:
                    approach = self.get_detector_approach(det_id)
                    approach_demand[approach] = detector_data[det_id]['count']
            
            # Adjust phase durations based on demand
            self.adjust_phase_timing(tl_id, approach_demand)
    
    def predictive_control(self, historical_data):
        # Use machine learning for traffic prediction
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare training data
        features = []  # Time, day of week, weather, etc.
        targets = []   # Traffic volumes
        
        # Train model
        model = RandomForestRegressor()
        model.fit(features, targets)
        
        # Make predictions
        current_features = self.extract_current_features()
        predicted_volume = model.predict([current_features])[0]
        
        # Adjust control strategies based on predictions
        self.proactive_signal_adjustment(predicted_volume)

# Usage example
smart_city = SmartCityManager()

while traci.simulation.getMinExpectedNumber() > 0:
    detector_data, tl_data = smart_city.collect_city_wide_data()
    smart_city.incident_detection(detector_data)
    smart_city.adaptive_signal_control(detector_data, tl_data)
    traci.simulationStep()
```

## 14. Troubleshooting and Best Practices {#troubleshooting}

### Common Issues and Solutions

#### Network Issues
```bash
# Check network connectivity
netconvert --check-lane-foes.all network.net.xml

# Fix common network problems
netconvert --osm-files input.osm -o output.net.xml \
           --remove-edges.isolated \
           --junctions.join \
           --tls.guess \
           --ramps.guess \
           --roundabouts.guess

# Validate network
sumo --net-file network.net.xml --route-files empty.rou.xml --begin 0 --end 1
```

#### Route and Demand Issues
```python
# Validate routes
import sumolib

net = sumolib.net.readNet('network.net.xml')
routes = sumolib.xml.parse('routes.rou.xml', 'route')

for route in routes:
    edges = route.edges.split()
    # Check if route is valid
    for i in range(len(edges)-1):
        edge1 = net.getEdge(edges[i])
        edge2 = net.getEdge(edges[i+1])
        
        # Check connectivity
        connections = edge1.getOutgoing()
        if edge2 not in connections:
            print(f"Invalid route: {route.id}, edges {edges[i]} -> {edges[i+1]}")
```

#### Performance Debugging
```xml
<!-- Enable detailed logging -->
<configuration>
    <report>
        <verbose value="true"/>
        <log value="simulation.log"/>
        <message-log value="messages.log"/>
        <error-log value="errors.log"/>
    </report>
    
    <processing>
        <step-length value="0.1"/>
        <collision.action value="warn"/>
        <time-to-teleport value="300"/>
    </processing>
</configuration>
```

### Best Practices

#### Simulation Design
1. **Start Simple**: Begin with basic scenarios and add complexity gradually
2. **Validate Early**: Check network and demand files before running long simulations
3. **Use Appropriate Detail**: Match model complexity to research questions
4. **Calibrate Models**: Use real-world data to calibrate vehicle and driver parameters
5. **Version Control**: Keep track of configuration changes

#### Code Organization
```python
# Recommended project structure
project/
├── config/
│   ├── network.net.xml
│   ├── routes.rou.xml
│   ├── additional.add.xml
│   └── simulation.sumocfg
├── scripts/
│   ├── generate_demand.py
│   ├── run_simulation.py
│   └── analyze_results.py
├── output/
│   ├── tripinfo.xml
│   ├── detector.xml
│   └── summary.xml
└── analysis/
    ├── plots/
    └── reports/
```

#### Performance Guidelines
1. **Network Optimization**: Remove unnecessary detail
2. **Demand Modeling**: Use flows instead of individual vehicles when possible
3. **Output Management**: Only collect necessary data
4. **Memory Management**: Clear unused data structures
5. **Parallel Processing**: Use multiple cores when available

### Debugging Techniques

#### Visual Debugging
```python
# Real-time debugging with TraCI
import traci

def debug_vehicle(veh_id):
    print(f"Vehicle {veh_id}:")
    print(f"  Position: {traci.vehicle.getPosition(veh_id)}")
    print(f"  Speed: {traci.vehicle.getSpeed(veh_id):.2f} m/s")
    print(f"  Lane: {traci.vehicle.getLaneID(veh_id)}")
    print(f"  Route: {traci.vehicle.getRoute(veh_id)}")
    print(f"  Waiting Time: {traci.vehicle.getWaitingTime(veh_id):.2f} s")
    print()

# Use in simulation loop
while traci.simulation.getMinExpectedNumber() > 0:
    vehicles = traci.vehicle.getIDList()
    if vehicles:
        debug_vehicle(vehicles[0])  # Debug first vehicle
    traci.simulationStep()
```

#### Data Validation
```python
# Validate simulation results
def validate_results(tripinfo_file):
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    
    issues = []
    
    for trip in root.findall('tripinfo'):
        duration = float(trip.get('duration'))
        waiting_time = float(trip.get('waitingTime'))
        time_loss = float(trip.get('timeLoss'))
        
        # Check for unrealistic values
        if duration < 0:
            issues.append(f"Negative duration for {trip.get('id')}")
        if waiting_time > duration:
            issues.append(f"Waiting time > duration for {trip.get('id')}")
        if time_loss > duration:
            issues.append(f"Time loss > duration for {trip.get('id')}")
    
    return issues

# Usage
issues = validate_results('tripinfo.xml')
if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Advanced Configuration Tips

#### Custom Output
```xml
<!-- Detailed output configuration -->
<additionalFile>
    <!-- Custom detectors -->
    <laneAreaDetector id="detector1" lane="edge1_0" pos="0" length="100" 
                      freq="30" file="lanearea.xml"/>
    
    <!-- Rerouters for dynamic assignment -->
    <rerouter id="rerouter1" edges="edge1 edge2" file="rerouter.xml">
        <interval begin="0" end="86400">
            <closingReroute id="edge3"/>
        </interval>
    </rerouter>
    
    <!-- Calibrators for flow adjustment -->
    <calibrator id="cal1" lane="edge1_0" pos="0" freq="60" 
                output="calibrator.xml" routeProbe="probe1"/>
</additionalFile>
```

#### Simulation Scenarios
```python
# Scenario management system
class ScenarioManager:
    def __init__(self):
        self.scenarios = {}
    
    def create_scenario(self, name, config):
        self.scenarios[name] = {
            'network': config.get('network', 'network.net.xml'),
            'demand': config.get('demand', 'routes.rou.xml'),
            'additional': config.get('additional', 'additional.add.xml'),
            'parameters': config.get('parameters', {}),
            'outputs': config.get('outputs', [])
        }
    
    def run_scenario(self, name):
        if name not in self.scenarios:
            raise ValueError(f"Scenario {name} not found")
        
        scenario = self.scenarios[name]
        
        # Create configuration file
        config_content = f'''
        <configuration>
            <input>
                <net-file value="{scenario['network']}"/>
                <route-files value="{scenario['demand']}"/>
                <additional-files value="{scenario['additional']}"/>
            </input>
            <processing>
                <step-length value="{scenario['parameters'].get('step_length', 1.0)}"/>
            </processing>
            <output>
                <tripinfo-output value="output/{name}_tripinfo.xml"/>
            </output>
        </configuration>
        '''
        
        with open(f'{name}.sumocfg', 'w') as f:
            f.write(config_content)
        
        # Run simulation
        traci.start(["sumo", "-c", f"{name}.sumocfg"])
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
        
        traci.close()
        
        return f"output/{name}_tripinfo.xml"

# Usage
manager = ScenarioManager()

# Define scenarios
manager.create_scenario('baseline', {
    'network': 'baseline_network.net.xml',
    'demand': 'baseline_demand.rou.xml',
    'parameters': {'step_length': 1.0}
})

manager.create_scenario('optimized', {
    'network': 'optimized_network.net.xml', 
    'demand': 'optimized_demand.rou.xml',
    'parameters': {'step_length': 0.5}
})

# Run scenarios
baseline_results = manager.run_scenario('baseline')
optimized_results = manager.run_scenario('optimized')
```

This comprehensive guide covers all aspects of SUMO from basic usage to advanced applications. You now have the knowledge to:

- Set up and configure SUMO simulations
- Create complex traffic scenarios
- Integrate SUMO with other tools and programming languages  
- Optimize simulation performance
- Apply SUMO to real-world transportation problems
- Debug and troubleshoot common issues

Practice with the examples provided, start with simple scenarios, and gradually build up to more complex simulations. The key to mastering SUMO is hands-on experience with real projects and continuous learning from the extensive SUMO documentation and community resources.
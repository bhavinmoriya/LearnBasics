# Complete TraCI Expert Guide
*From Beginner to Expert in Traffic Control Interface*

## Table of Contents
1. [Introduction to TraCI](#introduction)
2. [Setup and Installation](#setup)
3. [Basic Concepts](#basics)
4. [Core API Functions](#core-api)
5. [Vehicle Control](#vehicle-control)
6. [Traffic Light Control](#traffic-lights)
7. [Advanced Features](#advanced)
8. [Performance Optimization](#optimization)
9. [Real-World Applications](#applications)
10. [Expert Tips & Tricks](#expert-tips)

---

## 1. Introduction to TraCI {#introduction}

**TraCI (Traffic Control Interface)** is a TCP-based client-server architecture that allows external applications to access and control a running SUMO (Simulation of Urban Mobility) simulation. It enables real-time interaction with the traffic simulation, making it perfect for:

- Intelligent Transportation Systems (ITS)
- Adaptive traffic light control
- Vehicle routing optimization
- Traffic flow analysis
- Connected and autonomous vehicle research

### Key Features:
- **Real-time control**: Modify simulation during runtime
- **Multi-language support**: Python, C++, Java, MATLAB
- **Comprehensive API**: Control vehicles, traffic lights, detectors, routes
- **Scalability**: Handle large-scale traffic networks

---

## 2. Setup and Installation {#setup}

### Prerequisites:
```bash
# Install SUMO
sudo apt-get install sumo sumo-tools sumo-doc

# Install Python TraCI library
pip install traci sumolib
```

### Basic Project Structure:
```
project/
├── simulation.py      # Main TraCI script
├── network.net.xml    # Road network file
├── routes.rou.xml     # Vehicle routes
├── config.sumo.cfg    # SUMO configuration
└── data/              # Output data folder
```

### Minimal Setup Example:
```python
import traci
import sumolib

# Start SUMO with TraCI
sumo_cmd = ["sumo-gui", "-c", "config.sumo.cfg"]
traci.start(sumo_cmd)

# Main simulation loop
step = 0
while step < 1000:
    traci.simulationStep()
    step += 1

traci.close()
```

---

## 3. Basic Concepts {#basics}

### TraCI Communication Flow:
1. **Client** (your Python script) connects to **Server** (SUMO)
2. Commands sent via TCP socket
3. SUMO executes commands and returns results
4. Simulation advances step by step

### Key Objects in TraCI:
- **Vehicles**: Individual cars, trucks, buses
- **Routes**: Predefined paths through the network
- **Edges**: Road segments
- **Lanes**: Individual lanes within edges
- **Traffic Lights**: Signal controllers
- **Detectors**: Sensors for traffic measurement
- **POIs**: Points of Interest
- **Polygons**: Area definitions

### Coordinate Systems:
- **Network coordinates**: SUMO's internal coordinate system
- **Geo coordinates**: Real-world GPS coordinates
- **Lane coordinates**: Position along a specific lane

---

## 4. Core API Functions {#core-api}

### Simulation Control:
```python
# Start simulation
traci.start(sumo_cmd)

# Advance simulation by one step (1 second default)
traci.simulationStep()

# Advance by specific time
traci.simulationStep(5)  # Advance 5 seconds

# Get current simulation time
current_time = traci.simulation.getTime()

# Get loaded vehicles
loaded_vehicles = traci.simulation.getLoadedIDList()

# Get departed vehicles in current step
departed = traci.simulation.getDepartedIDList()

# Close connection
traci.close()
```

### Information Retrieval:
```python
# Get all vehicle IDs currently in simulation
vehicle_ids = traci.vehicle.getIDList()

# Get all traffic light IDs
tl_ids = traci.trafficlight.getIDList()

# Get all edge IDs
edge_ids = traci.edge.getIDList()

# Get simulation statistics
arrived_vehicles = traci.simulation.getArrivedNumber()
running_vehicles = traci.simulation.getMinExpectedNumber()
```

---

## 5. Vehicle Control {#vehicle-control}

### Adding Vehicles Dynamically:
```python
# Add vehicle with specific route
route_id = "route1"
vehicle_id = "vehicle_001"
traci.vehicle.add(vehicle_id, route_id, typeID="passenger")

# Add vehicle with specific parameters
traci.vehicle.add(
    vehID="custom_vehicle",
    routeID="route1",
    typeID="truck",
    depart="now",
    departLane="random",
    departPos="base",
    departSpeed="0"
)
```

### Vehicle State Information:
```python
vehicle_id = "vehicle_001"

# Position and movement
position = traci.vehicle.getPosition(vehicle_id)  # (x, y) coordinates
speed = traci.vehicle.getSpeed(vehicle_id)        # m/s
angle = traci.vehicle.getAngle(vehicle_id)        # degrees
acceleration = traci.vehicle.getAcceleration(vehicle_id)

# Road network position
edge_id = traci.vehicle.getRoadID(vehicle_id)
lane_id = traci.vehicle.getLaneID(vehicle_id)
lane_position = traci.vehicle.getLanePosition(vehicle_id)

# Traffic information
route = traci.vehicle.getRoute(vehicle_id)
route_index = traci.vehicle.getRouteIndex(vehicle_id)
distance_to_end = traci.vehicle.getDistance(vehicle_id)
```

### Vehicle Control Commands:
```python
vehicle_id = "vehicle_001"

# Speed control
traci.vehicle.setSpeed(vehicle_id, 15.0)      # Set speed to 15 m/s
traci.vehicle.slowDown(vehicle_id, 5.0, 10)   # Slow to 5 m/s over 10 seconds

# Lane changing
traci.vehicle.changeLane(vehicle_id, 1, 5)    # Change to lane 1 over 5 seconds

# Route modification
new_route = ["edge1", "edge2", "edge3"]
traci.vehicle.setRoute(vehicle_id, new_route)

# Stop commands
traci.vehicle.setStop(vehicle_id, "edge1", 100, duration=30)  # Stop for 30s

# Remove vehicle
traci.vehicle.remove(vehicle_id)
```

### Advanced Vehicle Control:
```python
# Car-following model parameters
traci.vehicle.setTau(vehicle_id, 1.5)         # Reaction time
traci.vehicle.setImperfection(vehicle_id, 0.2) # Driver imperfection

# Vehicle type modifications
traci.vehicle.setMaxSpeed(vehicle_id, 25.0)   # Max speed m/s
traci.vehicle.setLength(vehicle_id, 5.0)      # Vehicle length
traci.vehicle.setWidth(vehicle_id, 2.0)       # Vehicle width

# Color for visualization
traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))  # RGBA
```

---

## 6. Traffic Light Control {#traffic-lights}

### Basic Traffic Light Operations:
```python
tl_id = "junction1"

# Get current state
current_phase = traci.trafficlight.getPhase(tl_id)
current_program = traci.trafficlight.getProgram(tl_id)
phase_duration = traci.trafficlight.getPhaseDuration(tl_id)

# Get traffic light definition
logic = traci.trafficlight.getAllProgramLogics(tl_id)

# Set traffic light state
# State string: 'r' = red, 'g' = green, 'y' = yellow, 'G' = green priority
traci.trafficlight.setRedYellowGreenState(tl_id, "GGrrrrGGrrrr")

# Switch to next phase
traci.trafficlight.setPhase(tl_id, 2)

# Set phase duration
traci.trafficlight.setPhaseDuration(tl_id, 45.0)  # 45 seconds
```

### Adaptive Traffic Light Control:
```python
def adaptive_traffic_control(tl_id):
    """Implement simple adaptive control based on waiting vehicles"""
    
    # Get controlled lanes
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    
    waiting_times = {}
    vehicle_counts = {}
    
    for lane in controlled_lanes:
        # Get vehicles on each lane
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        vehicle_counts[lane] = len(vehicles)
        
        # Calculate total waiting time
        total_waiting = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
        waiting_times[lane] = total_waiting
    
    # Determine which phase should have priority
    current_phase = traci.trafficlight.getPhase(tl_id)
    
    # Simple logic: extend green phase if vehicles are waiting
    if max(waiting_times.values()) > 60:  # 60 seconds threshold
        # Extend current phase
        traci.trafficlight.setPhaseDuration(tl_id, 60)
    else:
        # Use default timing
        traci.trafficlight.setPhaseDuration(tl_id, 30)
```

### Custom Traffic Light Programs:
```python
# Create custom traffic light program
phases = []

# Phase 1: North-South green
phase1 = traci.trafficlight.Phase(45, "GGrrrrGGrrrr", 0, 0, [])
phases.append(phase1)

# Phase 2: Yellow transition
phase2 = traci.trafficlight.Phase(5, "yyrrrryyrrrr", 0, 0, [])
phases.append(phase2)

# Phase 3: East-West green
phase3 = traci.trafficlight.Phase(35, "rrGGGGrrGGGG", 0, 0, [])
phases.append(phase3)

# Phase 4: Yellow transition
phase4 = traci.trafficlight.Phase(5, "rryyyyyryyyy", 0, 0, [])
phases.append(phase4)

# Create and set the program
program_logic = traci.trafficlight.Logic("custom_program", 0, 0, phases)
traci.trafficlight.setProgramLogic(tl_id, program_logic)
traci.trafficlight.setProgram(tl_id, "custom_program")
```

---

## 7. Advanced Features {#advanced}

### Multi-Modal Transportation:
```python
# Add different vehicle types
traci.vehicle.add("bus_001", "bus_route", typeID="bus")
traci.vehicle.add("bike_001", "bike_route", typeID="bicycle")
traci.vehicle.add("pedestrian_001", "", typeID="pedestrian")

# Person simulation
person_id = "person_001"
traci.person.add(person_id, "edge1", 0)

# Add walking stage
traci.person.appendWalkingStage(person_id, ["edge1", "edge2"], 0)

# Add driving stage
traci.person.appendDrivingStage(person_id, "edge3", "car_001")
```

### Detector Usage:
```python
# Induction loop detectors
detector_id = "detector_001"

# Get detector measurements
vehicle_count = traci.inductionloop.getLastStepVehicleNumber(detector_id)
mean_speed = traci.inductionloop.getLastStepMeanSpeed(detector_id)
occupancy = traci.inductionloop.getLastStepOccupancy(detector_id)

# Get vehicles that passed
vehicles = traci.inductionloop.getLastStepVehicleIDs(detector_id)

# Multi-entry/exit detectors (E2)
e2_detector = "e2_001"
jam_length = traci.multientryexit.getLastStepMeanSpeed(e2_detector)
vehicle_number = traci.multientryexit.getLastStepVehicleNumber(e2_detector)
```

### Route and Edge Manipulation:
```python
# Dynamic route assignment based on traffic conditions
def dynamic_routing(vehicle_id, destination):
    current_edge = traci.vehicle.getRoadID(vehicle_id)
    
    # Get travel times for different routes
    route_options = [
        ["edge1", "edge2", "edge3", destination],
        ["edge1", "edge4", "edge5", destination],
        ["edge1", "edge6", "edge7", destination]
    ]
    
    best_route = None
    min_travel_time = float('inf')
    
    for route in route_options:
        total_time = 0
        for edge in route:
            travel_time = traci.edge.getTraveltime(edge)
            total_time += travel_time
        
        if total_time < min_travel_time:
            min_travel_time = total_time
            best_route = route
    
    # Assign best route
    traci.vehicle.setRoute(vehicle_id, best_route)

# Update edge weights based on current traffic
for edge_id in traci.edge.getIDList():
    current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
    max_speed = traci.edge.getMaxSpeed(edge_id)
    
    # Calculate congestion factor
    if current_speed > 0:
        congestion_factor = max_speed / current_speed
        new_travel_time = traci.edge.getLength(edge_id) / current_speed
        traci.edge.setEffort(edge_id, congestion_factor)
        traci.edge.adaptTraveltime(edge_id, new_travel_time)
```

### Data Collection and Analysis:
```python
import csv
from collections import defaultdict

class TrafficDataCollector:
    def __init__(self):
        self.vehicle_data = defaultdict(list)
        self.edge_data = defaultdict(list)
        self.tl_data = defaultdict(list)
    
    def collect_vehicle_data(self, step):
        for vehicle_id in traci.vehicle.getIDList():
            data = {
                'step': step,
                'vehicle_id': vehicle_id,
                'speed': traci.vehicle.getSpeed(vehicle_id),
                'position': traci.vehicle.getPosition(vehicle_id),
                'edge': traci.vehicle.getRoadID(vehicle_id),
                'lane': traci.vehicle.getLaneID(vehicle_id),
                'waiting_time': traci.vehicle.getWaitingTime(vehicle_id),
                'co2_emission': traci.vehicle.getCO2Emission(vehicle_id),
                'fuel_consumption': traci.vehicle.getFuelConsumption(vehicle_id)
            }
            self.vehicle_data[vehicle_id].append(data)
    
    def collect_edge_data(self, step):
        for edge_id in traci.edge.getIDList():
            data = {
                'step': step,
                'edge_id': edge_id,
                'vehicle_count': traci.edge.getLastStepVehicleNumber(edge_id),
                'mean_speed': traci.edge.getLastStepMeanSpeed(edge_id),
                'occupancy': traci.edge.getLastStepOccupancy(edge_id),
                'travel_time': traci.edge.getTraveltime(edge_id)
            }
            self.edge_data[edge_id].append(data)
    
    def export_to_csv(self, filename_prefix):
        # Export vehicle data
        with open(f"{filename_prefix}_vehicles.csv", 'w', newline='') as f:
            if self.vehicle_data:
                writer = csv.DictWriter(f, fieldnames=list(self.vehicle_data[list(self.vehicle_data.keys())[0]][0].keys()))
                writer.writeheader()
                for vehicle_records in self.vehicle_data.values():
                    writer.writerows(vehicle_records)
        
        # Export edge data
        with open(f"{filename_prefix}_edges.csv", 'w', newline='') as f:
            if self.edge_data:
                writer = csv.DictWriter(f, fieldnames=list(self.edge_data[list(self.edge_data.keys())[0]][0].keys()))
                writer.writeheader()
                for edge_records in self.edge_data.values():
                    writer.writerows(edge_records)
```

---

## 8. Performance Optimization {#optimization}

### Efficient TraCI Usage:
```python
# Batch operations when possible
vehicle_ids = traci.vehicle.getIDList()

# Instead of individual calls:
# for vid in vehicle_ids:
#     speed = traci.vehicle.getSpeed(vid)

# Use subscription for frequent data access
traci.vehicle.subscribe("vehicle_001", [traci.constants.VAR_SPEED, 
                                       traci.constants.VAR_POSITION,
                                       traci.constants.VAR_ROAD_ID])

# Get subscribed data
subscription_results = traci.vehicle.getSubscriptionResults("vehicle_001")
speed = subscription_results[traci.constants.VAR_SPEED]
position = subscription_results[traci.constants.VAR_POSITION]
```

### Context Subscriptions:
```python
# Subscribe to all vehicles in a specific area
traci.vehicle.subscribeContext("detector_001", 
                               traci.constants.CMD_GET_VEHICLE_VARIABLE,
                               100.0,  # radius in meters
                               [traci.constants.VAR_SPEED])

context_results = traci.vehicle.getContextSubscriptionResults("detector_001")
for vehicle_id, data in context_results.items():
    speed = data[traci.constants.VAR_SPEED]
```

### Memory Management:
```python
# Limit simulation output
traci.simulation.saveState("checkpoint.xml")  # Save state for later

# Clear finished vehicles from memory
for vehicle_id in traci.simulation.getArrivedIDList():
    # Process final data before vehicle is removed
    pass

# Use generators for large datasets
def vehicle_data_generator():
    for vehicle_id in traci.vehicle.getIDList():
        yield {
            'id': vehicle_id,
            'speed': traci.vehicle.getSpeed(vehicle_id),
            'position': traci.vehicle.getPosition(vehicle_id)
        }

# Process data incrementally
for vehicle_data in vehicle_data_generator():
    # Process individual vehicle data
    process_vehicle(vehicle_data)
```

---

## 9. Real-World Applications {#applications}

### Intelligent Traffic Signal Control:
```python
class AdaptiveTrafficController:
    def __init__(self, junction_id):
        self.junction_id = junction_id
        self.phase_history = []
        self.waiting_threshold = 50  # seconds
        
    def get_traffic_pressure(self):
        """Calculate traffic pressure on each approach"""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        pressures = {}
        
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
            queue_length = len([v for v in vehicles if traci.vehicle.getSpeed(v) < 0.5])
            
            # Combine waiting time and queue length
            pressures[lane] = total_waiting + (queue_length * 10)
            
        return pressures
    
    def decide_phase_timing(self):
        """Decide optimal phase timing based on traffic conditions"""
        pressures = self.get_traffic_pressure()
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        
        # Get lanes that have green in current phase
        state = traci.trafficlight.getRedYellowGreenState(self.junction_id)
        green_lanes = [lane for i, lane in enumerate(traci.trafficlight.getControlledLanes(self.junction_id))
                      if state[i] in ['G', 'g']]
        
        # Calculate pressure on green vs red approaches
        green_pressure = sum(pressures.get(lane, 0) for lane in green_lanes)
        total_pressure = sum(pressures.values())
        red_pressure = total_pressure - green_pressure
        
        # Extend green if green approach has higher pressure
        if green_pressure > red_pressure * 1.5:
            return min(60, traci.trafficlight.getPhaseDuration(self.junction_id) + 10)
        elif red_pressure > green_pressure * 2:
            return max(10, traci.trafficlight.getPhaseDuration(self.junction_id) - 5)
        else:
            return 30  # Default timing
```

### Connected Vehicle Communication:
```python
class V2XCommunication:
    def __init__(self, communication_range=300):
        self.range = communication_range
        self.message_buffer = {}
    
    def broadcast_traffic_info(self, sender_id):
        """Broadcast traffic information to nearby vehicles"""
        sender_pos = traci.vehicle.getPosition(sender_id)
        sender_speed = traci.vehicle.getSpeed(sender_id)
        sender_edge = traci.vehicle.getRoadID(sender_id)
        
        message = {
            'sender': sender_id,
            'position': sender_pos,
            'speed': sender_speed,
            'edge': sender_edge,
            'timestamp': traci.simulation.getTime(),
            'traffic_density': traci.edge.getLastStepVehicleNumber(sender_edge)
        }
        
        # Find vehicles in communication range
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id == sender_id:
                continue
                
            vehicle_pos = traci.vehicle.getPosition(vehicle_id)
            distance = ((sender_pos[0] - vehicle_pos[0])**2 + 
                       (sender_pos[1] - vehicle_pos[1])**2)**0.5
            
            if distance <= self.range:
                if vehicle_id not in self.message_buffer:
                    self.message_buffer[vehicle_id] = []
                self.message_buffer[vehicle_id].append(message)
    
    def process_received_messages(self, vehicle_id):
        """Process messages received by a vehicle"""
        if vehicle_id not in self.message_buffer:
            return
        
        messages = self.message_buffer[vehicle_id]
        
        # Process traffic density information
        edge_densities = {}
        for msg in messages:
            edge = msg['edge']
            density = msg['traffic_density']
            if edge not in edge_densities:
                edge_densities[edge] = []
            edge_densities[edge].append(density)
        
        # Use information for routing decisions
        current_route = traci.vehicle.getRoute(vehicle_id)
        current_index = traci.vehicle.getRouteIndex(vehicle_id)
        
        # Check if upcoming edges are congested
        for i in range(current_index + 1, len(current_route)):
            edge = current_route[i]
            if edge in edge_densities:
                avg_density = sum(edge_densities[edge]) / len(edge_densities[edge])
                if avg_density > 20:  # High density threshold
                    # Trigger re-routing
                    self.request_alternative_route(vehicle_id)
                    break
        
        # Clear processed messages
        self.message_buffer[vehicle_id] = []
```

### Emission and Fuel Optimization:
```python
class EcoRouting:
    def __init__(self):
        self.emission_factors = {
            'passenger': {'co2': 120, 'nox': 0.8, 'fuel': 6.5},  # g/km
            'truck': {'co2': 280, 'nox': 3.2, 'fuel': 22.0}
        }
    
    def calculate_route_emissions(self, route, vehicle_type):
        """Calculate total emissions for a given route"""
        total_emissions = {'co2': 0, 'nox': 0, 'fuel': 0}
        total_distance = 0
        
        for edge in route:
            edge_length = traci.edge.getLength(edge) / 1000  # Convert to km
            total_distance += edge_length
            
            # Get average speed on edge (affects emission factors)
            avg_speed = traci.edge.getLastStepMeanSpeed(edge) * 3.6  # m/s to km/h
            
            # Emission correction factor based on speed
            if avg_speed < 20:
                correction_factor = 1.3  # Higher emissions at low speeds
            elif avg_speed > 80:
                correction_factor = 1.2  # Higher emissions at high speeds
            else:
                correction_factor = 1.0
            
            for emission_type in total_emissions:
                base_emission = self.emission_factors[vehicle_type][emission_type]
                total_emissions[emission_type] += (base_emission * edge_length * correction_factor)
        
        return total_emissions, total_distance
    
    def find_eco_route(self, origin, destination, vehicle_type):
        """Find route with minimum environmental impact"""
        # Get possible routes (simplified - in practice use routing algorithm)
        possible_routes = self.get_possible_routes(origin, destination)
        
        best_route = None
        min_impact = float('inf')
        
        for route in possible_routes:
            emissions, distance = self.calculate_route_emissions(route, vehicle_type)
            
            # Calculate environmental impact score
            impact_score = (emissions['co2'] * 0.001 +  # Weight CO2
                           emissions['nox'] * 0.01 +   # Weight NOx
                           emissions['fuel'] * 0.002)  # Weight fuel consumption
            
            if impact_score < min_impact:
                min_impact = impact_score
                best_route = route
        
        return best_route, min_impact
```

---

## 10. Expert Tips & Tricks {#expert-tips}

### Advanced Debugging:
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# TraCI debug mode
traci.setOrder(1)  # Set execution order for multiple clients

# Monitor TraCI commands
def debug_traci_call(func, *args, **kwargs):
    """Wrapper to log TraCI calls"""
    print(f"TraCI call: {func.__name__} with args: {args}, kwargs: {kwargs}")
    result = func(*args, **kwargs)
    print(f"Result: {result}")
    return result

# Use with any TraCI function
speed = debug_traci_call(traci.vehicle.getSpeed, "vehicle_001")
```

### Error Handling Best Practices:
```python
def safe_traci_operation(operation, *args, **kwargs):
    """Safely execute TraCI operations with error handling"""
    try:
        return operation(*args, **kwargs)
    except traci.TraCIException as e:
        print(f"TraCI Error: {e}")
        return None
    except ConnectionError:
        print("Connection lost. Attempting to reconnect...")
        # Reconnection logic here
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
speed = safe_traci_operation(traci.vehicle.getSpeed, "vehicle_001")
if speed is not None:
    print(f"Vehicle speed: {speed}")
```

### Performance Monitoring:
```python
import time
from functools import wraps

def measure_performance(func):
    """Decorator to measure TraCI operation performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Apply to simulation step
@measure_performance
def simulation_step_with_control():
    traci.simulationStep()
    # Your control logic here
    for vehicle_id in traci.vehicle.getIDList():
        # Vehicle control operations
        pass

# Monitor step timing
for step in range(1000):
    simulation_step_with_control()
```

### Advanced Configuration:
```python
# Custom SUMO configuration for TraCI
sumo_config = {
    "--step-length": "0.1",        # 0.1 second time steps
    "--lateral-resolution": "0.8",  # Lane changing resolution
    "--collision.action": "warn",   # Handle collisions
    "--time-to-teleport": "300",   # Teleport stuck vehicles
    "--max-depart-delay": "600",   # Maximum departure delay
    "--routing-algorithm": "dijkstra",
    "--device.rerouting.period": "60",
    "--device.rerouting.adaptation-steps": "10"
}

sumo_cmd = ["sumo"] + [f"{key} {value}" for key, value in sumo_config.items()] + ["-c", "config.sumo.cfg"]
```

### Multi-Threading and Parallel Processing:
```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class ParallelTraCIController:
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def process_vehicles_parallel(self, vehicle_ids):
        """Process vehicle operations in parallel"""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            for vehicle_id in vehicle_ids:
                future = executor.submit(self.process_single_vehicle, vehicle_id)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing vehicle: {e}")
            
            return results
    
    def process_single_vehicle(self, vehicle_id):
        """Process individual vehicle (non-TraCI operations)"""
        # Perform CPU-intensive calculations here
        # Note: TraCI calls must be in main thread
        return {"vehicle_id": vehicle_id, "processed": True}
```

### Memory-Efficient Large Simulations:
```python
class MemoryEfficientSimulation:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        self.data_buffer = []
        
    def run_simulation(self, total_steps):
        """Run simulation with memory management"""
        for step in range(0, total_steps, self.chunk_size):
            chunk_end = min(step + self.chunk_size, total_steps)
            
            # Process chunk
            for current_step in range(step, chunk_end):
                traci.simulationStep()
                
                # Collect essential data only
                if current_step % 10 == 0:  # Sample every 10 steps
                    self.collect_sample_data(current_step)
            
            # Process and clear buffer
            self.process_data_chunk()
            self.data_buffer.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
    
# FinSim - Financial Simulation Framework

## Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Basic Test:**
   ```bash
   python test_finsim.py
   ```

3. **Run Simulation:**
   ```bash
   cd world_framework
   python simulator.py
   ```

## Components Fixed and Implemented

### Core Financial Simulation
- ✅ Environment management with thread-safe operations
- ✅ Economic instruments with price updates
- ✅ Trading agents with different strategies
- ✅ Broker agents for order matching
- ✅ Message bus for inter-agent communication

### Visualization
- ✅ Real-time plotting (when display available)
- ✅ Headless mode fallback for server environments
- ✅ Text-based market updates as fallback

### Issues Fixed
1. **world.py**: Fixed undefined variables, syntax errors, and incomplete implementations
2. **broker_agent.py**: Added missing Message import
3. **simulator.py**: Fixed critical loop condition bug and enabled visualization
4. **viz.py**: Added headless mode support and error handling
5. **requirements.txt**: Created proper dependency management

### Testing
- ✅ Basic functionality test (test_finsim.py)
- ✅ Headless simulator test (test_headless_simulator.py)
- ✅ Verified agents place orders correctly
- ✅ Verified broker processes trades
- ✅ Verified market updates work properly

## Usage

### Running with Visualization (if display available)
```python
from simulator import Simulator
simulator = Simulator(env, instruments_config, enable_visualization=True)
```

### Running in Headless Mode
```python
from simulator import Simulator
simulator = Simulator(env, instruments_config, enable_visualization=False)
```

The simulation framework is now fully functional with proper error handling and headless mode support.
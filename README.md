# FinSim

**FinSim** is a stock market and macro-economics simulation framework designed to provide a robust environment for educational purposes, research, and experimentation. It enables users to simulate a virtual economy where agents such as brokers, traders, and market instruments interact based on real-world economic principles. The framework is designed to be extensible, highly interactive, and capable of providing real-time visualizations of market dynamics.

---

## Features

- **Agent-Based Simulation:**
  - Supports multiple agent types such as brokers, heuristic traders, and statistical traders.
  - Agents interact with economic instruments and each other in real-time.

- **Economic Instrument Simulation:**
  - Simulates a variety of market instruments, including stocks, bonds, and commodities.
  - Instruments are configurable with parameters such as initial price, volatility, and historical price tracking.

- **Market and Macro-Economic Modeling:**
  - Models dynamic market conditions and economic trends.
  - Allows integration of custom economic rules and decision-making processes.

- **Real-Time Visualization:**
  - Provides real-time plotting of market conditions and instrument prices.
  - Includes support for external visualization tools like PlotJuggler via socket-based communication.

- **Extensibility:**
  - Easily add new agent types, instruments, or market rules.
  - Modular architecture to support user-specific enhancements.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/r3tr056/finsim.git
   cd finsim
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PlotJuggler (Optional):**
   PlotJuggler is a powerful visualization tool for real-time data.
   - [Download PlotJuggler](https://github.com/facontidavide/PlotJuggler) and follow the installation instructions for your platform.

4. **Run the Simulation:**
   ```bash
   python simulator.py
   ```

---

## Core Components

### 1. **Environment**
   - Manages the overall simulation state, including agents, instruments, and market updates.
   - Synchronizes agent actions and instrument price updates.

### 2. **Economic Instruments**
   - Represents stocks, bonds, and other financial instruments.
   - Each instrument tracks its historical price and updates based on market conditions.

### 3. **Agents**
   - **Broker Agents:** Facilitate trades between buyers and sellers.
   - **Heuristic Traders:** Follow predefined strategies like momentum or mean-reversion.
   - **Statistical Traders:** Use statistical methods to identify trading opportunities.

### 4. **Simulator**
   - Orchestrates the simulation by managing agents, instruments, and visualizations.
   - Provides real-time updates to connected visualization tools.

### 5. **Visualization**
   - Includes a real-time plotting interface using Matplotlib.
   - Supports external tools like PlotJuggler via socket communication.

---

## Usage

### Configuration
Define instruments and agents in the `simulator.py` file:

```python
# Instruments Configuration
instruments_config = {
    "StockA": {"initial_price": 100, "volatility": 0.02},
    "BondX": {"initial_price": 50, "volatility": 0.005},
}

# Agents Configuration
broker = BrokerAgent("Broker1", env)
trader1 = HeuristicTrader("Trader1", env, strategy="momentum")
trader2 = HeuristicTrader("Trader2", env, strategy="mean_reversion")
```

### Running the Simulation
Run the simulation for a specified duration:

```bash
python simulator.py
```

---

## Extending FinSim

### Adding New Agents
1. Create a new agent class inheriting from `BaseAgent`.
2. Implement the `run()` method to define the agent's behavior.
3. Register the agent with the environment:
   ```python
   new_agent = CustomAgent("AgentName", env)
   env.add_agent(new_agent)
   ```

### Adding New Instruments
1. Create a new instrument class inheriting from `EconomicInstrument`.
2. Define custom pricing or behavior methods.
3. Register the instrument with the environment:
   ```python
   new_instrument = CustomInstrument("InstrumentName", initial_price, volatility)
   env.add_instrument(new_instrument)
   ```

---

## Example Visualizations

1. **Real-Time Price Charts**:
   Displays the price trends of all instruments during the simulation.
2. **Agent Performance Metrics**:
   Tracks the performance of individual agents over time.
3. **Market Trends**:
   Shows macroeconomic trends and market-wide indicators.

---

## Contributing

We welcome contributions to enhance FinSim. Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

---

## License

FinSim is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Inspired by real-world stock markets and economic modeling.
- Built using Python, Matplotlib, and PlotJuggler for visualization.

---

## Contact

For questions or feedback, reach out to us at [your_email@example.com].
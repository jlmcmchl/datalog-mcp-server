# DataLog MCP Server

A Model Context Protocol (MCP) server that provides comprehensive access to WPILib datalog files for FRC robot telemetry analysis. This server enables AI agents to analyze robot performance data, debug issues, and extract insights from competition logs.

#### Full conversation with claude: https://claude.ai/share/b4cbe0c5-cbb7-480c-bba6-3a7120264a38

## Features

### Core Functionality
- **Load datalog files** - Parse WPILib binary datalog format
- **Signal discovery** - List, search, and explore available signals
- **Time-based queries** - Get data at specific timestamps or ranges
- **Bulk operations** - Efficiently process multiple signals at once
- **Statistical analysis** - Calculate min/max/mean/std for numeric signals

### 🆕 Struct Schema Support
- **Automatic struct parsing** - Decode complex WPILib structures (Pose2d, ChassisSpeeds, etc.)
- **Schema discovery** - Extract struct definitions from datalog files
- **Field extraction** - Access individual struct fields as virtual signals
- **Type-safe decoding** - Proper handling of nested structs and arrays
- **Built-in WPILib types** - Support for all common FRC data structures

### Advanced Analysis
- **Event detection** - Find robot events (mode changes, brownouts, etc.)
- **Match phase analysis** - Automatically detect auto/teleop/endgame periods  
- **Signal correlation** - Synchronize signals for comparative analysis
- **Data export** - Export to CSV for external analysis tools
- **Performance optimization** - Signal caching and resampling

### FRC-Specific Features
- **Robot event detection** - Mode changes, brownouts, CAN errors
- **Match data extraction** - Competition timing and FMS data
- **NetworkTables filtering** - Focus on NT-sourced signals
- **Signal hierarchy** - Organized view of robot subsystems

## Installation

### Prerequisites
```bash
pip install fastmcp wpiutil
```

### Files Required
- `datalog_manager.py` - Core datalog analysis engine
- `datalog_mcp_server.py` - MCP server implementation
- WPILib datalog files (`.wpilog` format)

## Usage

### Starting the Server
```bash
python datalog_mcp_server.py
```

### Basic Workflow
1. **Load a datalog file**
   ```python
   result = await datalog_load("FRC_20240315_143022_CMP_Q42.wpilog")
   ```

2. **Explore available signals**
   ```python
   signals = datalog_list_signals()
   drivetrain_signals = datalog_search_signals(r".*[Dd]rive.*")
   ```

3. **Analyze signal data**
   ```python
   stats = datalog_get_signal_statistics("/drivetrain/left_velocity")
   events = datalog_find_signal_events("/drivetrain/left_velocity", "value > 3.0")
   ```

4. **Export results**
   ```python
   datalog_export_to_csv(["signal1", "signal2"], "analysis.csv")
   ```

## Available Tools

### File Operations
- `datalog_load(filename)` - Load datalog file
- `datalog_get_log_info()` - File metadata and statistics

### Signal Discovery  
- `datalog_list_signals(pattern=None)` - List all signals
- `datalog_search_signals(pattern)` - Search by regex pattern
- `datalog_get_signal_hierarchy()` - Nested signal structure
- `datalog_get_network_table_signals()` - NT signals only

### Signal Information
- `datalog_get_signal_info(signal_name)` - Signal metadata
- `datalog_get_signal_metadata(signal_name)` - Custom metadata
- `datalog_get_signal_timespan(signal_name)` - Time range

### Data Retrieval
- `datalog_get_signal_value_at_timestamp(signal, time)` - Value at time
- `datalog_get_signal_value_in_range(signal, start, end)` - Range query
- `datalog_get_multiple_signals(signals, time_range=None)` - Bulk retrieval
- `datalog_get_all_signals_at_timestamp(time)` - Complete snapshot

### Data Analysis
- `datalog_get_signal_changes(signal, time_range=None)` - Only changes
- `datalog_find_signal_events(signal, condition)` - Conditional search
- `datalog_get_signal_statistics(signal, time_range=None)` - Statistics
- `datalog_synchronize_signals(signals)` - Align timestamps

### Robot Events
- `datalog_get_robot_events()` - Detected events
- `datalog_get_match_phases()` - Auto/teleop/endgame
- `datalog_find_match_data()` - Competition information

### Struct Schema Operations
- `datalog_get_struct_schemas()` - Get all struct type definitions
- `datalog_get_struct_schema(struct_name)` - Get specific struct schema
- `datalog_get_struct_signals()` - Find all signals using structs
- `datalog_get_signals_by_struct_type(struct_name)` - Find signals by type
- `datalog_get_struct_field_as_signal(signal, field_path)` - Extract struct fields

### Utilities
- `datalog_export_to_csv(signals, filename, time_range=None)` - CSV export
- `datalog_preload_signals(signals)` - Cache signals
- `datalog_resample_signal(signal, rate_hz)` - Resample data

## Example Use Cases

### Struct Data Analysis
```python
# Get available struct schemas
schemas = await datalog_get_struct_schemas()
print(f"Found {len(schemas)} struct types")

# Find all Pose2d signals
pose_signals = await datalog_get_signals_by_struct_type("Pose2d")

# Extract robot X position over time
x_position = await datalog_get_struct_field_as_signal(
    "/Robot//DriveState/Pose", 
    "translation.x"
)

# Extract swerve module speeds
module_speeds = await datalog_get_struct_field_as_signal(
    "/Robot//DriveState/ModuleStates",
    "0.speed"  # First module's speed
)
```

### Performance Analysis
```python
# Load match data
await datalog_load("match_log.wpilog")

# Find velocity signals
velocity_signals = datalog_search_signals(r".*velocity.*")

# Analyze autonomous performance  
phases = datalog_get_match_phases()
auto_phase = next(p for p in phases if p['phase'] == 'auto')

auto_data = datalog_get_signal_value_in_range(
    "/drivetrain/velocity", 
    auto_phase['start_time'], 
    auto_phase['end_time']
)
```

### Debug Issues
```python
# Find brownout events
voltage_events = datalog_find_signal_events(
    "/power/battery_voltage", 
    "value < 6.8"
)

# Check robot events during problems
robot_events = datalog_get_robot_events()
problem_events = [
    event for event in robot_events 
    if event['event_type'] in ['brownout', 'error', 'warning']
]
```

### Statistical Analysis
```python
# Compare motor performance across match
motor_signals = datalog_search_signals(r".*motor.*current.*")

for signal in motor_signals:
    stats = datalog_get_signal_statistics(signal)
    print(f"{signal}: avg={stats['mean']:.1f}A, max={stats['max']:.1f}A")

# Correlation analysis
sync_data = datalog_synchronize_signals(motor_signals)
datalog_export_to_csv(motor_signals, "motor_correlation.csv")
```

### Competition Analysis
```python
# Extract match information
match_data = datalog_find_match_data()
if match_data['fms_connected']:
    print("Competition match detected")

# Analyze by match phase
phases = datalog_get_match_phases()
for phase in phases:
    phase_stats = datalog_get_signal_statistics(
        "/drivetrain/velocity", 
        phase['start_time'], 
        phase['end_time']
    )
    print(f"{phase['phase']}: max speed = {phase_stats['max']:.1f} m/s")
```

## Supported WPILib Struct Types

The server automatically recognizes and parses these WPILib struct types:

| Struct Type | Fields | Description |
|-------------|--------|-------------|
| `Pose2d` | translation (Translation2d), rotation (Rotation2d) | Robot pose on field |
| `Pose3d` | translation (Translation3d), rotation (Rotation3d) | 3D robot pose |
| `Translation2d` | x (double), y (double) | 2D coordinates |
| `Translation3d` | x, y, z (double) | 3D coordinates |
| `Rotation2d` | value (double) | 2D rotation in radians |
| `Rotation3d` | quaternion (Quaternion) | 3D rotation |
| `Quaternion` | w, x, y, z (double) | Quaternion components |
| `ChassisSpeeds` | vx, vy, omega (double) | Robot velocity |
| `SwerveModuleState` | speed (double), angle (Rotation2d) | Swerve module target |
| `SwerveModulePosition` | distance (double), angle (Rotation2d) | Swerve module position |

### Struct Field Access

Use dot notation to access nested fields:

```python
# Robot position components
x_pos = await datalog_get_struct_field_as_signal("/Robot//DriveState/Pose", "translation.x")
y_pos = await datalog_get_struct_field_as_signal("/Robot//DriveState/Pose", "translation.y")
heading = await datalog_get_struct_field_as_signal("/Robot//DriveState/Pose", "rotation.value")

# Chassis velocity components  
vx = await datalog_get_struct_field_as_signal("/Robot//DriveState/Speeds", "vx")
vy = await datalog_get_struct_field_as_signal("/Robot//DriveState/Speeds", "vy")
omega = await datalog_get_struct_field_as_signal("/Robot//DriveState/Speeds", "omega")

# Individual swerve module data (for arrays, use index)
module0_speed = await datalog_get_struct_field_as_signal("/Robot//DriveState/ModuleStates", "0.speed")
module0_angle = await datalog_get_struct_field_as_signal("/Robot//DriveState/ModuleStates", "0.angle.value")
```

## Resources

The server provides browsable resources for data exploration:

- `datalog://signals` - List all available signals
- `datalog://hierarchy` - Signal organization structure  
- `datalog://events` - Robot events timeline
- `datalog://match` - Match phases and competition data
- `datalog://structs` - 🆕 Struct schemas and type information
- `datalog://{signal_name}` - Individual signal data and metadata

## Data Format

### Signal Information
```json
{
  "name": "/drivetrain/left_velocity",
  "type": "double",
  "metadata": {"units": "m/s", "source": "encoder"},
  "first_timestamp": 0.0,
  "last_timestamp": 150.0,
  "record_count": 7500,
  "entry_id": 42
}
```

### Signal Values
```json
{
  "timestamp": 45.123,
  "value": 2.34,
  "valid": true
}
```

### Robot Events
```json
{
  "timestamp": 15.0,
  "event_type": "mode_change", 
  "description": "Robot mode changed to teleop",
  "data": {"mode": "teleop", "previous_mode": "auto"}
}
```

### Match Phases
```json
{
  "phase": "auto",
  "start_time": 0.0,
  "end_time": 15.0,
  "duration": 15.0
}
```

## Advanced Features

### Signal Conditions
Use natural language conditions in `datalog_find_signal_events()`:
- `"value > 3.0"` - Values above threshold
- `"value < 0.5"` - Values below threshold  
- `"value >= 2.0"` - Greater than or equal
- `"value != 0"` - Non-zero values

### Time Range Filtering
Most analysis functions accept optional time ranges:
```python
# Analyze only autonomous period
auto_stats = datalog_get_signal_statistics(
    "/drivetrain/velocity",
    start_time=0.0,
    end_time=15.0
)
```

### Signal Patterns
Use regex patterns for flexible signal discovery:
- `r".*velocity.*"` - All velocity signals
- `r"/drivetrain/.*"` - All drivetrain signals
- `r".*current$"` - Signals ending with "current"
- `r"^/vision/.*"` - Vision subsystem signals

## Performance Tips

1. **Preload frequently used signals**
   ```python
   datalog_preload_signals(["/drivetrain/velocity", "/power/battery_voltage"])
   ```

2. **Use time ranges to limit data**
   ```python
   # Only get teleop data
   teleop_data = datalog_get_signal_value_in_range(signal, 15.0, 135.0)
   ```

3. **Export large datasets**
   ```python
   # For extensive analysis, export to CSV
   datalog_export_to_csv(all_signals, "full_match.csv")
   ```

4. **Resample high-frequency signals**
   ```python
   # Reduce 200Hz signal to 20Hz for analysis
   resampled = datalog_resample_signal("/high_freq_sensor", 20.0)
   ```

## Error Handling

The server provides detailed error information:
- Invalid signal names return `None` or empty lists
- File loading errors include descriptive messages
- Statistics functions handle non-numeric data gracefully
- Time range queries validate bounds automatically

## Compatibility

- **WPILib Versions**: 2023.4.3+ (binary datalog format)
- **Python**: 3.8+
- **FRC Season**: 2023+ (when binary datalogs introduced)
- **File Formats**: `.wpilog` files only

## Contributing

To extend the server:

1. **Add new analysis methods** to `DataLogManager` class
2. **Create MCP tool wrappers** in the server file
3. **Update documentation** with new capabilities
4. **Add usage examples** for new features

## Troubleshooting

### Common Issues

**"Invalid datalog file"**
- Ensure file is `.wpilog` format (not CSV)
- Check file isn't corrupted or incomplete
- Verify WPILib version compatibility

**"Signal not found"**  
- Use `datalog_list_signals()` to see available signals
- Check signal name spelling and case
- Some signals may only exist during certain periods

**"No data in time range"**
- Verify time range with `datalog_get_signal_timespan()`
- Check match phases with `datalog_get_match_phases()`
- Signal may not have been active during that period

**Memory usage with large files**
- Use time ranges to limit data scope
- Preload only necessary signals
- Export large datasets rather than keeping in memory

## License

This project is released under the MIT License, making it freely available for FRC teams and educational use.
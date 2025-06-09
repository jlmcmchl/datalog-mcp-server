#!/usr/bin/env python3
"""
DataLog MCP Server Usage Example

This example demonstrates how to use the DataLog MCP server to analyze
robot telemetry data from WPILib datalog files.
"""

import asyncio
import json
from typing import List, Dict, Any

# Example usage of the DataLog MCP tools
# (This would be called from an MCP client)


async def analyze_robot_performance():
    """Example analysis workflow using DataLog MCP tools"""

    # 1. Load a datalog file
    print("Loading datalog file...")
    load_result = await datalog_load("FRC_20240315_143022_CMP_Q42.wpilog")
    if not load_result["success"]:
        print(f"Failed to load datalog: {load_result.get('error')}")
        return

    print(f"Loaded datalog with {load_result['log_info']['signal_count']} signals")
    print(f"Duration: {load_result['log_info']['duration']:.2f} seconds")

    # 2. Explore available signals
    print("\nDiscovering signals...")
    all_signals = datalog_list_signals()
    print(f"Found {len(all_signals)} total signals")

    # Find drivetrain-related signals
    drivetrain_signals = datalog_search_signals(r".*[Dd]rive.*")
    print(f"Found {len(drivetrain_signals)} drivetrain signals:")
    for signal in drivetrain_signals[:5]:  # Show first 5
        print(f"  - {signal}")

    # 3. Get signal hierarchy for better understanding
    hierarchy = datalog_get_signal_hierarchy()
    print(f"\nSignal hierarchy has {len(hierarchy)} top-level categories")

    # 4. Analyze specific signals
    if "/drivetrain/left_velocity" in all_signals:
        print("\nAnalyzing left drivetrain velocity...")

        # Get signal information
        signal_info = datalog_get_signal_info("/drivetrain/left_velocity")
        print(f"Signal type: {signal_info['type']}")
        print(f"Records: {signal_info['record_count']}")
        print(
            f"Time span: {signal_info['first_timestamp']:.2f} - {signal_info['last_timestamp']:.2f}s"
        )

        # Get statistics
        stats = datalog_get_signal_statistics("/drivetrain/left_velocity")
        if stats:
            print(f"Statistics:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Std Dev: {stats['stdev']:.2f}")

    # 5. Find events of interest
    print("\nLooking for high-speed events...")
    if "/drivetrain/left_velocity" in all_signals:
        high_speed_events = datalog_find_signal_events(
            "/drivetrain/left_velocity", "value > 3.0"
        )
        print(f"Found {len(high_speed_events)} high-speed events")

        if high_speed_events:
            for event in high_speed_events[:3]:  # Show first 3
                print(f"  {event['timestamp']:.2f}s: {event['value']:.2f} m/s")

    # 6. Analyze match phases
    print("\nAnalyzing match phases...")
    match_phases = datalog_get_match_phases()
    for phase in match_phases:
        print(
            f"{phase['phase'].upper()}: {phase['start_time']:.1f}s - {phase['end_time']:.1f}s ({phase['duration']:.1f}s)"
        )

    # 7. Look for robot events
    robot_events = datalog_get_robot_events()
    print(f"\nFound {len(robot_events)} robot events:")
    for event in robot_events[:5]:  # Show first 5
        print(
            f"  {event['timestamp']:.2f}s: {event['event_type']} - {event['description']}"
        )

    # 8. Multi-signal analysis
    print("\nAnalyzing multiple signals together...")
    motor_signals = [
        s for s in all_signals if "motor" in s.lower() and "current" in s.lower()
    ]
    if motor_signals:
        print(f"Found {len(motor_signals)} motor current signals")

        # Get synchronized data for correlation analysis
        if len(motor_signals) >= 2:
            sync_data = datalog_synchronize_signals(motor_signals[:2])
            print(f"Synchronized {len(sync_data)} signals")

    # 9. Export data for external analysis
    print("\nExporting data...")
    export_signals = [s for s in all_signals if "velocity" in s.lower()][
        :5
    ]  # First 5 velocity signals
    if export_signals:
        export_result = datalog_export_to_csv(export_signals, "velocity_analysis.csv")
        if export_result["success"]:
            print(
                f"Exported {export_result['signal_count']} signals to {export_result['filename']}"
            )

    print("\nAnalysis complete!")


async def debug_autonomous_performance():
    """Example: Debug autonomous performance using match phase data"""

    print("Analyzing autonomous performance...")

    # Get match phases to focus on autonomous
    phases = datalog_get_match_phases()
    auto_phase = next((p for p in phases if p["phase"] == "auto"), None)

    if not auto_phase:
        print("No autonomous phase detected")
        return

    print(
        f"Autonomous phase: {auto_phase['start_time']:.1f}s - {auto_phase['end_time']:.1f}s"
    )

    # Get robot position data during auto
    position_signals = datalog_search_signals(r".*[Pp]osition.*")
    if position_signals:
        print(f"Found {len(position_signals)} position signals")

        # Analyze movement during auto only
        auto_position_data = datalog_get_signal_value_in_range(
            position_signals[0], auto_phase["start_time"], auto_phase["end_time"]
        )

        if auto_position_data:
            start_pos = auto_position_data[0]["value"]
            end_pos = auto_position_data[-1]["value"]
            print(f"Robot moved from {start_pos} to {end_pos} during autonomous")

    # Look for errors during autonomous
    auto_events = [
        event
        for event in datalog_get_robot_events()
        if auto_phase["start_time"] <= event["timestamp"] <= auto_phase["end_time"]
    ]

    if auto_events:
        print(f"Found {len(auto_events)} events during autonomous:")
        for event in auto_events:
            print(f"  {event['timestamp']:.1f}s: {event['description']}")
    else:
        print("No events detected during autonomous (good!)")


async def battery_analysis():
    """Example: Analyze battery performance throughout match"""

    print("Analyzing battery performance...")

    # Look for battery voltage signals
    voltage_signals = datalog_search_signals(r".*[Bb]attery.*[Vv]oltage.*")
    if not voltage_signals:
        voltage_signals = datalog_search_signals(r".*[Vv]oltage.*")

    if not voltage_signals:
        print("No voltage signals found")
        return

    voltage_signal = voltage_signals[0]
    print(f"Using signal: {voltage_signal}")

    # Get voltage statistics
    stats = datalog_get_signal_statistics(voltage_signal)
    if stats:
        print(f"Voltage statistics:")
        print(f"  Mean: {stats['mean']:.2f}V")
        print(f"  Min: {stats['min']:.2f}V")
        print(f"  Max: {stats['max']:.2f}V")

    # Find brownout events (voltage drops below 6.8V)
    brownout_events = datalog_find_signal_events(voltage_signal, "value < 6.8")
    if brownout_events:
        print(f"\nWARNING: Found {len(brownout_events)} potential brownout events!")
        for event in brownout_events[:5]:
            print(f"  {event['timestamp']:.1f}s: {event['value']:.2f}V")
    else:
        print("\nNo brownout events detected (good!)")

    # Analyze voltage during different match phases
    phases = datalog_get_match_phases()
    for phase in phases:
        phase_stats = datalog_get_signal_statistics(
            voltage_signal, phase["start_time"], phase["end_time"]
        )
        if phase_stats:
            print(
                f"{phase['phase'].upper()} voltage: {phase_stats['mean']:.2f}V avg, {phase_stats['min']:.2f}V min"
            )


def print_available_tools():
    """Print all available DataLog MCP tools"""

    tools = [
        "datalog_load - Load a WPILib datalog file",
        "datalog_list_signals - List all signal names",
        "datalog_search_signals - Find signals by pattern",
        "datalog_get_signal_info - Get signal metadata",
        "datalog_get_signal_hierarchy - Get nested signal structure",
        "datalog_get_signal_timespan - Get signal time range",
        "datalog_get_signal_value_at_timestamp - Get value at specific time",
        "datalog_get_signal_value_in_range - Get values in time range",
        "datalog_get_multiple_signals - Bulk signal retrieval",
        "datalog_get_all_signals_at_timestamp - Snapshot at specific time",
        "datalog_get_signal_changes - Only changed values",
        "datalog_find_signal_events - Find values meeting criteria",
        "datalog_get_signal_statistics - Calculate signal statistics",
        "datalog_synchronize_signals - Align signals to common timestamps",
        "datalog_export_to_csv - Export to CSV file",
        "datalog_preload_signals - Cache signals in memory",
        "datalog_get_log_info - Overall log information",
        "datalog_get_robot_events - Detected robot events",
        "datalog_find_match_data - Competition match information",
        "datalog_get_match_phases - Auto/teleop/endgame phases",
        "datalog_get_network_table_signals - NetworkTables signals only",
        "datalog_resample_signal - Downsample/interpolate signal",
    ]

    print("Available DataLog MCP Tools:")
    print("=" * 50)
    for tool in tools:
        print(f"• {tool}")

    print(f"\nTotal: {len(tools)} tools available")


# Example workflows that could be triggered by AI agents
EXAMPLE_WORKFLOWS = {
    "performance_analysis": analyze_robot_performance,
    "autonomous_debug": debug_autonomous_performance,
    "battery_analysis": battery_analysis,
}


if __name__ == "__main__":
    print("DataLog MCP Server Usage Examples")
    print("=" * 40)

    print_available_tools()

    print("\nExample workflows:")
    for name, func in EXAMPLE_WORKFLOWS.items():
        print(f"• {name}: {func.__doc__}")

    print("\nTo run the server:")
    print("python datalog_mcp_server.py")

    print("\nTo use with Claude or other MCP clients:")
    print("Configure the MCP client to connect to this server and use the tools above.")

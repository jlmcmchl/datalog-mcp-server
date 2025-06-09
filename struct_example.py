#!/usr/bin/env python3
"""
Struct Schema Support Usage Example

This example demonstrates how to use the enhanced DataLog MCP server
with struct schema support for analyzing complex WPILib data structures.
"""

import asyncio
from typing import List, Dict, Any


async def analyze_robot_structs():
    """Example analysis workflow using struct data"""

    # 1. Get all available struct schemas
    print("Getting struct schemas...")
    schemas = await datalog_get_struct_schemas()

    print(f"Found {len(schemas)} struct schemas:")
    for name, schema in schemas.items():
        fields_str = ", ".join([f"{f['name']}:{f['type']}" for f in schema["fields"]])
        print(f"  {name}: {{{fields_str}}}")

    # 2. Find signals that use struct types
    print("\nFinding struct signals...")
    struct_signals = await datalog_get_struct_signals()
    print(f"Found {len(struct_signals)} signals using struct types:")

    for signal in struct_signals[:5]:  # Show first 5
        signal_info = await datalog_get_signal_info(signal)
        print(f"  {signal} -> {signal_info['type']}")

    # 3. Analyze specific struct types
    print("\nAnalyzing Pose2d signals...")
    pose_signals = await datalog_get_signals_by_struct_type("Pose2d")
    print(f"Found {len(pose_signals)} Pose2d signals:")
    for signal in pose_signals:
        print(f"  {signal}")

    # 4. Extract specific fields from struct data
    if "/Robot//DriveState/Pose" in pose_signals:
        print("\nExtracting robot X position over time...")
        x_position = await datalog_get_struct_field_as_signal(
            "/Robot//DriveState/Pose", "translation.x"
        )

        if x_position:
            print(f"Got {len(x_position)} X position samples")
            # Show first few and last few samples
            for i in [0, 1, 2, -3, -2, -1]:
                if 0 <= i < len(x_position) or -len(x_position) <= i < 0:
                    sample = x_position[i]
                    print(f"  {sample['timestamp']:.2f}s: X = {sample['value']:.3f}m")

            # Get statistics for X position
            x_values = [s["value"] for s in x_position if s["value"] is not None]
            if x_values:
                print(f"  X Position Stats:")
                print(f"    Min: {min(x_values):.3f}m")
                print(f"    Max: {max(x_values):.3f}m")
                print(f"    Range: {max(x_values) - min(x_values):.3f}m")

    # 5. Analyze SwerveModuleState array
    print("\nAnalyzing SwerveModuleState data...")
    module_signals = await datalog_get_signals_by_struct_type("SwerveModuleState[]")

    if module_signals:
        print(f"Found {len(module_signals)} SwerveModuleState array signals:")
        for signal in module_signals:
            print(f"  {signal}")

        # Get a sample of module states
        if "/Robot//DriveState/ModuleStates" in module_signals:
            module_data = await datalog_get_signal_value_in_range(
                "/Robot//DriveState/ModuleStates",
                20.0,  # Start at 20 seconds
                25.0,  # End at 25 seconds
            )

            if module_data:
                print(f"\nModule states sample (20-25s): {len(module_data)} samples")
                # Show first sample structure
                first_sample = module_data[0]
                print(f"Sample at {first_sample['timestamp']:.2f}s:")
                print(f"  Value type: {type(first_sample['value'])}")
                if isinstance(first_sample["value"], list):
                    print(f"  Number of modules: {len(first_sample['value'])}")
                    if first_sample["value"]:
                        module = first_sample["value"][0]
                        print(f"  Module 0 structure: {type(module)}")

    # 6. Analyze ChassisSpeeds
    print("\nAnalyzing ChassisSpeeds data...")
    speeds_signals = await datalog_get_signals_by_struct_type("ChassisSpeeds")

    if speeds_signals:
        for signal in speeds_signals:
            print(f"Found ChassisSpeeds signal: {signal}")

            # Extract individual velocity components
            vx_data = await datalog_get_struct_field_as_signal(signal, "vx")
            vy_data = await datalog_get_struct_field_as_signal(signal, "vy")
            omega_data = await datalog_get_struct_field_as_signal(signal, "omega")

            if vx_data and vy_data and omega_data:
                print(f"  Extracted velocity components:")
                print(f"    VX samples: {len(vx_data)}")
                print(f"    VY samples: {len(vy_data)}")
                print(f"    Omega samples: {len(omega_data)}")

                # Calculate max speeds
                max_vx = max([s["value"] for s in vx_data if s["value"] is not None])
                max_vy = max([s["value"] for s in vy_data if s["value"] is not None])
                max_omega = max(
                    [abs(s["value"]) for s in omega_data if s["value"] is not None]
                )

                print(f"    Max VX: {max_vx:.2f} m/s")
                print(f"    Max VY: {max_vy:.2f} m/s")
                print(f"    Max Omega: {max_omega:.2f} rad/s")


async def compare_odometry_vs_vision():
    """Compare robot pose from odometry vs vision"""

    print("Comparing odometry vs vision pose estimation...")

    # Get odometry pose
    odom_pose = await datalog_get_struct_field_as_signal(
        "/Robot//DriveState/Pose", "translation"
    )

    # Look for vision pose signals
    vision_signals = await datalog_search_signals(".*Vision.*Pose.*")
    print(f"Found {len(vision_signals)} vision-related pose signals")

    for signal in vision_signals[:3]:  # Check first 3
        signal_info = await datalog_get_signal_info(signal)
        print(f"  {signal} -> {signal_info['type']}")

    # If we have both, we could do correlation analysis
    if odom_pose:
        print(f"Odometry pose has {len(odom_pose)} samples")

        # Extract X and Y coordinates
        odom_x = await datalog_get_struct_field_as_signal(
            "/Robot//DriveState/Pose", "translation.x"
        )
        odom_y = await datalog_get_struct_field_as_signal(
            "/Robot//DriveState/Pose", "translation.y"
        )

        if odom_x and odom_y:
            print("Robot trajectory (odometry):")

            # Show trajectory at key points
            sample_indices = [
                0,
                len(odom_x) // 4,
                len(odom_x) // 2,
                3 * len(odom_x) // 4,
                -1,
            ]

            for i in sample_indices:
                if 0 <= i < len(odom_x) or -len(odom_x) <= i < 0:
                    x_sample = odom_x[i]
                    y_sample = odom_y[i]
                    print(
                        f"  {x_sample['timestamp']:.1f}s: ({x_sample['value']:.2f}, {y_sample['value']:.2f})"
                    )


async def analyze_swerve_module_performance():
    """Analyze individual swerve module performance"""

    print("Analyzing swerve module performance...")

    # Get module states (array of SwerveModuleState)
    module_states = await datalog_get_signal_value_in_range(
        "/Robot//DriveState/ModuleStates", 100.0, 150.0  # Analyze from 100s  # to 150s
    )

    if not module_states:
        print("No module state data found in range")
        return

    print(f"Analyzing {len(module_states)} module state samples")

    # Process module data (this would be done in the enhanced datalog manager)
    module_speeds = [[] for _ in range(4)]  # Assume 4 modules
    module_angles = [[] for _ in range(4)]

    for sample in module_states:
        if isinstance(sample["value"], list) and len(sample["value"]) >= 4:
            for i, module in enumerate(sample["value"][:4]):
                if isinstance(module, dict):
                    speed = module.get("speed")
                    angle = (
                        module.get("angle", {}).get("value")
                        if isinstance(module.get("angle"), dict)
                        else None
                    )

                    if speed is not None:
                        module_speeds[i].append(speed)
                    if angle is not None:
                        module_angles[i].append(angle)

    # Analyze each module
    for i in range(4):
        if module_speeds[i]:
            max_speed = max(module_speeds[i])
            avg_speed = sum(module_speeds[i]) / len(module_speeds[i])
            print(
                f"  Module {i}: Max speed = {max_speed:.2f} m/s, Avg = {avg_speed:.2f} m/s"
            )


# Example workflows
STRUCT_WORKFLOWS = {
    "struct_analysis": analyze_robot_structs,
    "odometry_vs_vision": compare_odometry_vs_vision,
    "swerve_analysis": analyze_swerve_module_performance,
}


def print_struct_capabilities():
    """Print the new struct-related capabilities"""

    capabilities = [
        "datalog_get_struct_schemas - Get all struct type definitions",
        "datalog_get_struct_schema - Get specific struct schema",
        "datalog_get_struct_signals - Find all signals using structs",
        "datalog_get_signals_by_struct_type - Find signals by struct type",
        "datalog_get_struct_field_as_signal - Extract struct fields as signals",
    ]

    print("New Struct Schema Capabilities:")
    print("=" * 50)
    for capability in capabilities:
        print(f"• {capability}")

    print(f"\nTotal: {len(capabilities)} new struct-related tools")
    print("\nSupported WPILib Struct Types:")

    struct_types = [
        "Pose2d - Robot pose (translation + rotation)",
        "Pose3d - 3D pose with quaternion rotation",
        "Translation2d - X,Y coordinates",
        "Translation3d - X,Y,Z coordinates",
        "Rotation2d - 2D rotation (radians)",
        "Rotation3d - 3D rotation (quaternion)",
        "ChassisSpeeds - Vx, Vy, Omega velocities",
        "SwerveModuleState - Speed + angle per module",
        "SwerveModulePosition - Distance + angle per module",
        "Quaternion - W,X,Y,Z quaternion components",
    ]

    for struct_type in struct_types:
        print(f"  • {struct_type}")


if __name__ == "__main__":
    print("DataLog Struct Schema Support")
    print("=" * 40)

    print_struct_capabilities()

    print("\nExample Analysis Workflows:")
    for name, func in STRUCT_WORKFLOWS.items():
        print(f"• {name}: {func.__doc__}")

    print("\nBenefits of Struct Support:")
    print("• Parse complex WPILib data structures automatically")
    print("• Extract individual fields as virtual signals")
    print("• Understand robot pose, velocity, and module states")
    print("• Correlate odometry with vision measurements")
    print("• Analyze swerve drive module performance")
    print("• Export structured data for external analysis")

#!/usr/bin/env python3
"""
DataLog Model Context Protocol (MCP) Server

This server provides MCP access to WPILib datalog files, enabling AI agents
to analyze robot telemetry data through a standardized protocol.
"""

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datalog_manager import (
    DataLogManager,
    SignalInfo,
    SignalValue,
    RobotEvent,
    MatchPhase,
    LogInfo,
)

from fastmcp import FastMCP, Context

# Configure logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def datalog_context(server: FastMCP) -> AsyncIterator[DataLogManager]:
    datalog_manager = DataLogManager()
    try:
        yield datalog_manager
    finally:
        # Cleanup if needed
        pass


mcp = FastMCP("DataLog MCP Server", lifespan=datalog_context)


@mcp.tool
def datalog_get_struct_schemas(ctx: Context = None) -> Dict[str, Dict[str, Any]]:
    """Get all available struct schemas"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    schemas = datalog_manager.get_struct_schemas()

    # Convert to serializable format
    result = {}
    for name, schema in schemas.items():
        result[name] = {
            "name": schema.name,
            "fields": [
                {"name": field.name, "type": field.type} for field in schema.fields
            ],
        }

    return result


@mcp.tool
def datalog_get_struct_schema(
    struct_name: str, ctx: Context = None
) -> Optional[Dict[str, Any]]:
    """Get a specific struct schema"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    schema = datalog_manager.get_struct_schema(struct_name)

    if schema:
        return {
            "name": schema.name,
            "fields": [
                {"name": field.name, "type": field.type} for field in schema.fields
            ],
        }
    return None


@mcp.tool
def datalog_get_struct_signals(ctx: Context = None) -> List[str]:
    """Get all signals that use struct types"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.get_struct_signals()


@mcp.tool
def datalog_get_signals_by_struct_type(
    struct_name: str, ctx: Context = None
) -> List[str]:
    """Get all signals that use a specific struct type"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.get_signals_by_struct_type(struct_name)


@mcp.tool
def datalog_get_struct_field_as_signal(
    signal_name: str, field_path: str, ctx: Context = None
) -> List[Dict[str, Any]]:
    """Extract a specific field from struct data as a virtual signal"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    field_values = datalog_manager.get_struct_field_as_signal(
        unquote(signal_name), field_path
    )

    return [
        {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
        for sv in field_values
    ]


@mcp.tool
async def datalog_load(
    filename: str,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Load a WPILib datalog file"""
    try:
        datalog_manager: DataLogManager = ctx.request_context.lifespan_context
        success = datalog_manager.load_datalog(filename)

        if success:
            log_info = datalog_manager.get_log_info()
            return {
                "success": True,
                "log_info": {
                    "filename": log_info.filename,
                    "file_size": log_info.file_size,
                    "duration": log_info.duration,
                    "start_time": log_info.start_time,
                    "end_time": log_info.end_time,
                    "total_records": log_info.total_records,
                    "signal_count": log_info.signal_count,
                    "robot_events_count": len(log_info.robot_events),
                },
            }
        else:
            return {"success": False, "error": "Failed to load datalog file"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def datalog_list_signals(
    pattern: Optional[str] = None, ctx: Context = None
) -> List[str]:
    """List all signal names, optionally filtered by pattern"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.list_signals(pattern)


@mcp.tool
def datalog_search_signals(pattern: str, ctx: Context = None) -> List[str]:
    """Find signals matching regex/glob patterns"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.search_signals(pattern)


@mcp.tool
def datalog_get_signal_info(
    signal_name: str, ctx: Context = None
) -> Optional[Dict[str, Any]]:
    """Get detailed information about a signal"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    signal_info = datalog_manager.get_signal_info(unquote(signal_name))

    if signal_info:
        return {
            "name": signal_info.name,
            "type": signal_info.type,
            "metadata": signal_info.metadata,
            "first_timestamp": signal_info.first_timestamp,
            "last_timestamp": signal_info.last_timestamp,
            "record_count": signal_info.record_count,
            "entry_id": signal_info.entry_id,
        }
    return None


@mcp.tool
def datalog_get_signal_metadata(
    signal_name: str, ctx: Context = None
) -> Dict[str, Any]:
    """Get metadata associated with a signal"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.get_signal_metadata(unquote(signal_name))


@mcp.tool
def datalog_get_signal_hierarchy(ctx: Context = None) -> Dict[str, Any]:
    """Return nested structure of signals"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.get_signal_hierarchy()


@mcp.tool
def datalog_get_signal_timespan(
    signal_name: str, ctx: Context = None
) -> Optional[Tuple[float, float]]:
    """Get the time span (start, end) for a signal"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.get_signal_timespan(unquote(signal_name))


@mcp.tool
def datalog_get_signal_value_at_timestamp(
    signal_name: str, timestamp: float, ctx: Context = None
) -> Optional[Dict[str, Any]]:
    """Get signal value at or nearest to a specific timestamp"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    signal_value = datalog_manager.get_signal_value_at_timestamp(
        unquote(signal_name), timestamp
    )

    if signal_value:
        return {
            "timestamp": signal_value.timestamp,
            "value": signal_value.value,
            "valid": signal_value.valid,
        }
    return None


@mcp.tool
def datalog_get_signal_value_in_range(
    signal_name: str, start_time: float, end_time: float, ctx: Context = None
) -> List[Dict[str, Any]]:
    """Get all signal values within a timestamp range"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    signal_values = datalog_manager.get_signal_value_in_range(
        unquote(signal_name), start_time, end_time
    )

    return [
        {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
        for sv in signal_values
    ]


@mcp.tool
def datalog_get_multiple_signals(
    signal_names: List[str],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    ctx: Context = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Efficiently fetch multiple signals at once"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context

    # Decode signal names
    decoded_names = [unquote(name) for name in signal_names]
    time_range = (
        (start_time, end_time)
        if start_time is not None and end_time is not None
        else None
    )

    signals_data = datalog_manager.get_multiple_signals(decoded_names, time_range)

    # Convert to serializable format
    result = {}
    for signal_name, values in signals_data.items():
        result[signal_name] = [
            {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
            for sv in values
        ]

    return result


@mcp.tool
def datalog_get_all_signals_at_timestamp(
    timestamp: float, ctx: Context = None
) -> Dict[str, Dict[str, Any]]:
    """Get snapshot of all active signals at a specific time"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    signals_snapshot = datalog_manager.get_all_signals_at_timestamp(timestamp)

    return {
        signal_name: {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
        for signal_name, sv in signals_snapshot.items()
    }


@mcp.tool
def datalog_get_signal_changes(
    signal_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    ctx: Context = None,
) -> List[Dict[str, Any]]:
    """Return only timestamps where signal value actually changed"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context

    time_range = (
        (start_time, end_time)
        if start_time is not None and end_time is not None
        else None
    )
    changes = datalog_manager.get_signal_changes(unquote(signal_name), time_range)

    return [
        {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
        for sv in changes
    ]


@mcp.tool
def datalog_find_signal_events(
    signal_name: str, condition: str, ctx: Context = None
) -> List[Dict[str, Any]]:
    """Find timestamps where signal meets criteria (e.g., 'value > 2.0')"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    events = datalog_manager.find_signal_events(unquote(signal_name), condition)

    return [
        {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
        for sv in events
    ]


@mcp.tool
def datalog_get_signal_statistics(
    signal_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    ctx: Context = None,
) -> Dict[str, float]:
    """Calculate statistics for numeric signals"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context

    time_range = (
        (start_time, end_time)
        if start_time is not None and end_time is not None
        else None
    )
    return datalog_manager.get_signal_statistics(unquote(signal_name), time_range)


@mcp.tool
def datalog_synchronize_signals(
    signal_names: List[str], ctx: Context = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Align signals to common timestamps for correlation analysis"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context

    decoded_names = [unquote(name) for name in signal_names]
    synchronized_data = datalog_manager.synchronize_signals(decoded_names)

    # Convert to serializable format
    result = {}
    for signal_name, values in synchronized_data.items():
        result[signal_name] = [
            {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
            for sv in values
        ]

    return result


@mcp.tool
def datalog_export_to_csv(
    signal_names: List[str],
    filename: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Export signals to CSV file"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context

    decoded_names = [unquote(name) for name in signal_names]
    time_range = (
        (start_time, end_time)
        if start_time is not None and end_time is not None
        else None
    )

    success = datalog_manager.export_to_csv(decoded_names, filename, time_range)

    return {
        "success": success,
        "filename": filename,
        "signal_count": len(decoded_names),
    }


@mcp.tool
def datalog_preload_signals(
    signal_names: List[str], ctx: Context = None
) -> Dict[str, Any]:
    """Cache frequently accessed signals in memory"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context

    decoded_names = [unquote(name) for name in signal_names]
    success = datalog_manager.preload_signals(decoded_names)

    return {"success": success, "preloaded_count": len(decoded_names)}


@mcp.tool
def datalog_get_log_info(ctx: Context = None) -> Optional[Dict[str, Any]]:
    """Get overall log file information"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    log_info = datalog_manager.get_log_info()

    if log_info:
        return {
            "filename": log_info.filename,
            "file_size": log_info.file_size,
            "duration": log_info.duration,
            "start_time": log_info.start_time,
            "end_time": log_info.end_time,
            "total_records": log_info.total_records,
            "signal_count": log_info.signal_count,
            "robot_events": [
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "description": event.description,
                    "data": event.data,
                }
                for event in log_info.robot_events
            ],
        }
    return None


@mcp.tool
def datalog_get_robot_events(ctx: Context = None) -> List[Dict[str, Any]]:
    """Get robot events detected in the log"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    events = datalog_manager.get_robot_events()

    return [
        {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "description": event.description,
            "data": event.data,
        }
        for event in events
    ]


@mcp.tool
def datalog_find_match_data(ctx: Context = None) -> Dict[str, Any]:
    """Detect competition match info if logged"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.find_match_data()


@mcp.tool
def datalog_get_match_phases(ctx: Context = None) -> List[Dict[str, Any]]:
    """Get detected match phases"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    phases = datalog_manager.get_match_phases()

    return [
        {
            "phase": phase.phase,
            "start_time": phase.start_time,
            "end_time": phase.end_time,
            "duration": phase.duration,
        }
        for phase in phases
    ]


@mcp.tool
def datalog_get_network_table_signals(ctx: Context = None) -> List[str]:
    """Filter to only NetworkTables-sourced data"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return datalog_manager.get_network_table_signals()


@mcp.tool
def datalog_resample_signal(
    signal_name: str, new_rate_hz: float, ctx: Context = None
) -> List[Dict[str, Any]]:
    """Downsample/interpolate signal for analysis"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    resampled = datalog_manager.resample_signal(unquote(signal_name), new_rate_hz)

    return [
        {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
        for sv in resampled
    ]


# Resources for browsing datalog structure
@mcp.resource("datalog://signals", mime_type="application/json")
def list_datalog_signals(ctx: Context = None) -> Dict[str, Any]:
    """List all datalog signals as resources"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    signals = datalog_manager.list_signals()

    resources = []
    for signal_name in signals:
        signal_info = datalog_manager.get_signal_info(signal_name)
        if signal_info:
            resources.append(
                {
                    "uri": f"datalog://{quote(signal_name)}",
                    "name": signal_name,
                    "description": f"Signal: {signal_info.type} ({signal_info.record_count} records)",
                    "mimeType": "application/json",
                }
            )

    return {"resources": resources}


@mcp.resource("datalog://hierarchy", mime_type="application/json")
def get_datalog_hierarchy(ctx: Context = None) -> Dict[str, Any]:
    """Get signal hierarchy as a resource"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    return {"hierarchy": datalog_manager.get_signal_hierarchy()}


@mcp.resource("datalog://events", mime_type="application/json")
def get_datalog_events(ctx: Context = None) -> Dict[str, Any]:
    """Get robot events as a resource"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    events = datalog_manager.get_robot_events()

    return {
        "events": [
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "description": event.description,
                "data": event.data,
            }
            for event in events
        ]
    }


@mcp.resource("datalog://structs", mime_type="application/json")
def get_datalog_structs(ctx: Context = None) -> Dict[str, Any]:
    """Get struct schemas as a resource"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    schemas = datalog_manager.get_struct_schemas()
    struct_signals = datalog_manager.get_struct_signals()

    return {
        "schemas": {
            name: {
                "name": schema.name,
                "fields": [
                    {"name": field.name, "type": field.type} for field in schema.fields
                ],
            }
            for name, schema in schemas.items()
        },
        "struct_signals": struct_signals,
        "signal_counts_by_type": {
            name: len(datalog_manager.get_signals_by_struct_type(name))
            for name in schemas.keys()
        },
    }


@mcp.resource("datalog://match", mime_type="application/json")
def get_datalog_match_info(ctx: Context = None) -> Dict[str, Any]:
    """Get match information as a resource"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    match_data = datalog_manager.find_match_data()
    phases = datalog_manager.get_match_phases()

    return {
        "match_data": match_data,
        "phases": [
            {
                "phase": phase.phase,
                "start_time": phase.start_time,
                "end_time": phase.end_time,
                "duration": phase.duration,
            }
            for phase in phases
        ],
    }


@mcp.resource("datalog://{signal_name}", mime_type="application/json")
def get_datalog_signal(signal_name: str, ctx: Context = None) -> Dict[str, Any]:
    """Get signal data as a resource"""
    datalog_manager: DataLogManager = ctx.request_context.lifespan_context
    decoded_name = unquote(signal_name)

    signal_info = datalog_manager.get_signal_info(decoded_name)
    if not signal_info:
        return {"error": f"Signal {decoded_name} not found"}

    # Get a sample of recent data (last 100 points to avoid huge responses)
    all_values = datalog_manager.signal_data.get(decoded_name, [])
    sample_values = all_values[-100:] if len(all_values) > 100 else all_values

    return {
        "signal_info": {
            "name": signal_info.name,
            "type": signal_info.type,
            "metadata": signal_info.metadata,
            "first_timestamp": signal_info.first_timestamp,
            "last_timestamp": signal_info.last_timestamp,
            "record_count": signal_info.record_count,
            "entry_id": signal_info.entry_id,
        },
        "sample_data": [
            {"timestamp": sv.timestamp, "value": sv.value, "valid": sv.valid}
            for sv in sample_values
        ],
        "sample_note": f"Showing last {len(sample_values)} of {len(all_values)} total records",
    }


def main():
    """Main entry point"""
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()

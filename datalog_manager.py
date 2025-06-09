#!/usr/bin/env python3
"""
DataLog Manager Module

This module provides a high-level interface for reading and analyzing WPILib datalog files
in FRC robotics applications. It handles file loading, signal discovery, time-based queries,
and provides analysis helpers for robot data.
"""

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import statistics
import csv
from pathlib import Path

from wpiutil.log import DataLogReader, DataLogRecord


logger = logging.getLogger(__name__)


@dataclass
class SignalInfo:
    """Information about a signal in the datalog"""

    name: str
    type: str
    metadata: Dict[str, Any]
    first_timestamp: float
    last_timestamp: float
    record_count: int
    entry_id: int


@dataclass
class SignalValue:
    """A timestamped signal value"""

    timestamp: float
    value: Any
    valid: bool = True


@dataclass
class RobotEvent:
    """Robot event detected in the log"""

    timestamp: float
    event_type: str
    description: str
    data: Dict[str, Any]


@dataclass
class MatchPhase:
    """Match phase timing information"""

    phase: str  # "auto", "teleop", "endgame", "disabled"
    start_time: float
    end_time: float
    duration: float


@dataclass
class LogInfo:
    """Overall datalog file information"""

    filename: str
    file_size: int
    duration: float
    start_time: float
    end_time: float
    total_records: int
    signal_count: int
    robot_events: List[RobotEvent]


class DataLogManager:
    """High-level interface for WPILib datalog analysis"""

    def __init__(self):
        self.reader = None
        self.filename = None
        self.signals: Dict[str, SignalInfo] = {}
        self.signal_data: Dict[str, List[SignalValue]] = defaultdict(list)
        self.robot_events: List[RobotEvent] = []
        self.match_phases: List[MatchPhase] = []
        self.log_info: Optional[LogInfo] = None
        self._preloaded_signals: set = set()
        self.access_lock = threading.Lock()

    def load_datalog(self, filename: str) -> bool:
        """Load a datalog file and parse its contents"""
        try:
            with self.access_lock:
                self.reader = DataLogReader(filename)
                if not self.reader.isValid():
                    logger.error(f"Invalid datalog file: {filename}")
                    return False

                self.filename = filename
                self._parse_log()
                self._detect_robot_events()
                self._detect_match_phases()
                self._generate_log_info()

                logger.info(f"Successfully loaded datalog: {filename}")
                return True

        except Exception as e:
            logger.error(f"Error loading datalog {filename}: {e}")
            return False

    def _parse_log(self):
        """Parse the datalog and build signal information"""
        entries = {}  # entry_id -> signal info

        for record in self.reader:
            if record.isStart():
                start_data = record.getStartData()
                entry_id = start_data.entry
                signal_name = start_data.name
                signal_type = start_data.type

                try:
                    metadata = (
                        json.loads(start_data.metadata) if start_data.metadata else {}
                    )
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": start_data.metadata}

                signal_info = SignalInfo(
                    name=signal_name,
                    type=signal_type,
                    metadata=metadata,
                    first_timestamp=float("inf"),
                    last_timestamp=float("-inf"),
                    record_count=0,
                    entry_id=entry_id,
                )

                entries[entry_id] = signal_info
                self.signals[signal_name] = signal_info

            elif not record.isControl():
                # Data record
                entry_id = record.getEntry()
                timestamp = record.getTimestamp() / 1e6  # Convert to seconds

                if entry_id in entries:
                    signal_info = entries[entry_id]
                    signal_info.first_timestamp = min(
                        signal_info.first_timestamp, timestamp
                    )
                    signal_info.last_timestamp = max(
                        signal_info.last_timestamp, timestamp
                    )
                    signal_info.record_count += 1

                    # Decode value based on type
                    value = self._decode_record_value(record, signal_info.type)

                    signal_value = SignalValue(
                        timestamp=timestamp, value=value, valid=True
                    )

                    self.signal_data[signal_info.name].append(signal_value)

    def _decode_record_value(self, record: DataLogRecord, signal_type: str) -> Any:
        """Decode a record value based on its type"""
        try:
            if signal_type == "boolean":
                return record.getBoolean()
            elif signal_type == "int64":
                return record.getInteger()
            elif signal_type == "float" or signal_type == "double":
                return record.getDouble()
            elif signal_type == "string":
                return record.getString()
            elif signal_type.endswith("[]"):
                # Array types
                base_type = signal_type[:-2]
                if base_type == "boolean":
                    return record.getBooleanArray()
                elif base_type == "int64":
                    return record.getIntegerArray()
                elif base_type == "double":
                    return record.getDoubleArray()
                elif base_type == "string":
                    return record.getStringArray()
            else:
                # Raw data for unknown types
                return record.getRaw()
        except Exception as e:
            logger.warning(f"Error decoding record of type {signal_type}: {e}")
            return None

    def _detect_robot_events(self):
        """Detect robot events from known signal patterns"""
        self.robot_events = []

        # Look for DS connection events
        if "/FMSInfo/FMSControlData" in self.signals:
            # Detect match start/end from FMS data
            pass

        # Look for robot mode changes
        if "/DriverStation/RobotMode" in self.signal_data:
            prev_mode = None
            for signal_value in self.signal_data["/DriverStation/RobotMode"]:
                if prev_mode != signal_value.value:
                    event = RobotEvent(
                        timestamp=signal_value.timestamp,
                        event_type="mode_change",
                        description=f"Robot mode changed to {signal_value.value}",
                        data={"mode": signal_value.value, "previous_mode": prev_mode},
                    )
                    self.robot_events.append(event)
                    prev_mode = signal_value.value

        # Look for brownout events
        if "/DriverStation/BrownedOut" in self.signal_data:
            for signal_value in self.signal_data["/DriverStation/BrownedOut"]:
                if signal_value.value:
                    event = RobotEvent(
                        timestamp=signal_value.timestamp,
                        event_type="brownout",
                        description="Robot brownout detected",
                        data={},
                    )
                    self.robot_events.append(event)

    def _detect_match_phases(self):
        """Detect match phases from robot mode data"""
        self.match_phases = []

        if "/DriverStation/RobotMode" not in self.signal_data:
            return

        current_phase = None
        phase_start = None

        for signal_value in self.signal_data["/DriverStation/RobotMode"]:
            mode = signal_value.value
            timestamp = signal_value.timestamp

            if mode == "auto" and current_phase != "auto":
                if current_phase and phase_start:
                    self._finish_phase(current_phase, phase_start, timestamp)
                current_phase = "auto"
                phase_start = timestamp

            elif mode == "teleop" and current_phase != "teleop":
                if current_phase and phase_start:
                    self._finish_phase(current_phase, phase_start, timestamp)
                current_phase = "teleop"
                phase_start = timestamp

            elif mode == "disabled" and current_phase != "disabled":
                if current_phase and phase_start:
                    self._finish_phase(current_phase, phase_start, timestamp)
                current_phase = "disabled"
                phase_start = timestamp

        # Finish the last phase
        if current_phase and phase_start and self.log_info:
            self._finish_phase(current_phase, phase_start, self.log_info.end_time)

    def _finish_phase(self, phase: str, start_time: float, end_time: float):
        """Helper to finish a match phase"""
        match_phase = MatchPhase(
            phase=phase,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
        self.match_phases.append(match_phase)

    def _generate_log_info(self):
        """Generate overall log information"""
        if not self.filename:
            return

        file_path = Path(self.filename)
        file_size = file_path.stat().st_size if file_path.exists() else 0

        start_time = (
            min(info.first_timestamp for info in self.signals.values())
            if self.signals
            else 0
        )
        end_time = (
            max(info.last_timestamp for info in self.signals.values())
            if self.signals
            else 0
        )
        total_records = sum(info.record_count for info in self.signals.values())

        self.log_info = LogInfo(
            filename=self.filename,
            file_size=file_size,
            duration=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
            total_records=total_records,
            signal_count=len(self.signals),
            robot_events=self.robot_events,
        )

    def list_signals(self, pattern: Optional[str] = None) -> List[str]:
        """List all signal names, optionally filtered by pattern"""
        signal_names = list(self.signals.keys())

        if pattern:
            regex = re.compile(pattern)
            signal_names = [name for name in signal_names if regex.search(name)]

        return sorted(signal_names)

    def search_signals(self, pattern: str) -> List[str]:
        """Find signals matching regex/glob patterns"""
        return self.list_signals(pattern)

    def get_signal_info(self, signal_name: str) -> Optional[SignalInfo]:
        """Get detailed information about a signal"""
        return self.signals.get(signal_name)

    def get_signal_metadata(self, signal_name: str) -> Dict[str, Any]:
        """Get metadata associated with a signal"""
        signal_info = self.signals.get(signal_name)
        return signal_info.metadata if signal_info else {}

    def get_signal_hierarchy(self) -> Dict[str, Any]:
        """Return nested structure of signals"""
        hierarchy = {}

        for signal_name in self.signals.keys():
            parts = signal_name.strip("/").split("/")
            current = hierarchy

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Leaf node
            if parts:
                current[parts[-1]] = {"_signal": True, "_name": signal_name}

        return hierarchy

    def get_signal_timespan(self, signal_name: str) -> Optional[Tuple[float, float]]:
        """Get the time span (start, end) for a signal"""
        signal_info = self.signals.get(signal_name)
        if signal_info:
            return (signal_info.first_timestamp, signal_info.last_timestamp)
        return None

    def get_signal_value_at_timestamp(
        self, signal_name: str, timestamp: float
    ) -> Optional[SignalValue]:
        """Get signal value at or nearest to a specific timestamp"""
        if signal_name not in self.signal_data:
            return None

        values = self.signal_data[signal_name]
        if not values:
            return None

        # Find closest timestamp
        closest_value = min(values, key=lambda v: abs(v.timestamp - timestamp))
        return closest_value

    def get_signal_value_in_range(
        self, signal_name: str, start_time: float, end_time: float
    ) -> List[SignalValue]:
        """Get all signal values within a timestamp range"""
        if signal_name not in self.signal_data:
            return []

        values = self.signal_data[signal_name]
        return [v for v in values if start_time <= v.timestamp <= end_time]

    def get_multiple_signals(
        self, signal_names: List[str], time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, List[SignalValue]]:
        """Efficiently fetch multiple signals at once"""
        result = {}

        for signal_name in signal_names:
            if time_range:
                result[signal_name] = self.get_signal_value_in_range(
                    signal_name, time_range[0], time_range[1]
                )
            else:
                result[signal_name] = self.signal_data.get(signal_name, [])

        return result

    def get_all_signals_at_timestamp(self, timestamp: float) -> Dict[str, SignalValue]:
        """Get snapshot of all active signals at a specific time"""
        result = {}

        for signal_name in self.signals.keys():
            value = self.get_signal_value_at_timestamp(signal_name, timestamp)
            if value:
                result[signal_name] = value

        return result

    def get_signal_changes(
        self, signal_name: str, time_range: Optional[Tuple[float, float]] = None
    ) -> List[SignalValue]:
        """Return only timestamps where signal value actually changed"""
        if signal_name not in self.signal_data:
            return []

        values = self.signal_data[signal_name]
        if time_range:
            values = [
                v for v in values if time_range[0] <= v.timestamp <= time_range[1]
            ]

        if not values:
            return []

        changes = [values[0]]  # Always include first value
        prev_value = values[0].value

        for value in values[1:]:
            if value.value != prev_value:
                changes.append(value)
                prev_value = value.value

        return changes

    def find_signal_events(self, signal_name: str, condition: str) -> List[SignalValue]:
        """Find timestamps where signal meets criteria"""
        if signal_name not in self.signal_data:
            return []

        values = self.signal_data[signal_name]
        events = []

        for value in values:
            try:
                # Simple condition evaluation - in production, use a safer eval alternative
                if self._evaluate_condition(value.value, condition):
                    events.append(value)
            except Exception as e:
                logger.warning(
                    f"Error evaluating condition '{condition}' for value {value.value}: {e}"
                )

        return events

    def _evaluate_condition(self, value: Any, condition: str) -> bool:
        """Safely evaluate a condition string"""
        # This is a simplified implementation - in production, use a proper expression parser
        try:
            # Replace 'value' in condition with actual value
            safe_condition = condition.replace("value", str(value))
            # Only allow basic operations
            allowed_chars = set("0123456789.+-*/()< >= ! = &|")
            if not all(c in allowed_chars or c.isspace() for c in safe_condition):
                return False
            return eval(safe_condition)
        except:
            return False

    def get_signal_statistics(
        self, signal_name: str, time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """Calculate statistics for numeric signals"""
        values = (
            self.get_signal_value_in_range(signal_name, time_range[0], time_range[1])
            if time_range
            else self.signal_data.get(signal_name, [])
        )

        if not values:
            return {}

        numeric_values = []
        for v in values:
            if isinstance(v.value, (int, float)):
                numeric_values.append(v.value)

        if not numeric_values:
            return {}

        return {
            "count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "stdev": (
                statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
            ),
        }

    def synchronize_signals(
        self, signal_names: List[str]
    ) -> Dict[str, List[SignalValue]]:
        """Align signals to common timestamps for correlation analysis"""
        if not signal_names:
            return {}

        # Get all timestamps from all signals
        all_timestamps = set()
        for signal_name in signal_names:
            if signal_name in self.signal_data:
                all_timestamps.update(
                    v.timestamp for v in self.signal_data[signal_name]
                )

        timestamps = sorted(all_timestamps)

        # Interpolate each signal to common timestamps
        result = {}
        for signal_name in signal_names:
            result[signal_name] = []
            for timestamp in timestamps:
                value = self.get_signal_value_at_timestamp(signal_name, timestamp)
                if value:
                    result[signal_name].append(value)
                else:
                    # Create placeholder for missing data
                    result[signal_name].append(
                        SignalValue(timestamp=timestamp, value=None, valid=False)
                    )

        return result

    def export_to_csv(
        self,
        signal_names: List[str],
        filename: str,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """Export signals to CSV file"""
        try:
            synchronized_data = self.synchronize_signals(signal_names)

            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Header
                header = ["timestamp"] + signal_names
                writer.writerow(header)

                # Get common timestamps
                if signal_names and signal_names[0] in synchronized_data:
                    timestamps = [
                        v.timestamp for v in synchronized_data[signal_names[0]]
                    ]

                    for i, timestamp in enumerate(timestamps):
                        if time_range and not (
                            time_range[0] <= timestamp <= time_range[1]
                        ):
                            continue

                        row = [timestamp]
                        for signal_name in signal_names:
                            if i < len(synchronized_data[signal_name]):
                                value = synchronized_data[signal_name][i].value
                                row.append(value if value is not None else "")
                            else:
                                row.append("")
                        writer.writerow(row)

            logger.info(f"Exported {len(signal_names)} signals to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV {filename}: {e}")
            return False

    def preload_signals(self, signal_names: List[str]) -> bool:
        """Cache frequently accessed signals in memory"""
        try:
            for signal_name in signal_names:
                if signal_name in self.signals:
                    # Data is already loaded during parsing, just mark as preloaded
                    self._preloaded_signals.add(signal_name)
            return True
        except Exception as e:
            logger.error(f"Error preloading signals: {e}")
            return False

    def get_log_info(self) -> Optional[LogInfo]:
        """Get overall log file information"""
        return self.log_info

    def get_robot_events(self) -> List[RobotEvent]:
        """Get robot events detected in the log"""
        return self.robot_events

    def find_match_data(self) -> Dict[str, Any]:
        """Detect competition match info if logged"""
        match_data = {}

        # Look for FMS data
        fms_signals = [
            name for name in self.signals.keys() if "FMS" in name or "Match" in name
        ]
        if fms_signals:
            match_data["fms_connected"] = True
            match_data["fms_signals"] = fms_signals
        else:
            match_data["fms_connected"] = False

        # Add match phases
        match_data["phases"] = [asdict(phase) for phase in self.match_phases]

        return match_data

    def get_network_table_signals(self) -> List[str]:
        """Filter to only NetworkTables-sourced data"""
        nt_signals = []
        for signal_name, signal_info in self.signals.items():
            # Check metadata for NT source indicator
            if signal_info.metadata.get(
                "source"
            ) == "NetworkTables" or signal_name.startswith("/NT/"):
                nt_signals.append(signal_name)
        return nt_signals

    def get_match_phases(self) -> List[MatchPhase]:
        """Get detected match phases"""
        return self.match_phases

    def resample_signal(
        self, signal_name: str, new_rate_hz: float
    ) -> List[SignalValue]:
        """Downsample/interpolate signal for analysis"""
        if signal_name not in self.signal_data:
            return []

        original_values = self.signal_data[signal_name]
        if not original_values:
            return []

        start_time = original_values[0].timestamp
        end_time = original_values[-1].timestamp
        duration = end_time - start_time

        if duration <= 0:
            return original_values

        # Generate new timestamps at specified rate
        sample_interval = 1.0 / new_rate_hz
        new_timestamps = []
        current_time = start_time

        while current_time <= end_time:
            new_timestamps.append(current_time)
            current_time += sample_interval

        # Interpolate values at new timestamps
        resampled = []
        for timestamp in new_timestamps:
            interpolated_value = self.get_signal_value_at_timestamp(
                signal_name, timestamp
            )
            if interpolated_value:
                resampled.append(
                    SignalValue(
                        timestamp=timestamp,
                        value=interpolated_value.value,
                        valid=interpolated_value.valid,
                    )
                )

        return resampled

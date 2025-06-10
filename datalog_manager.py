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
import struct
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
class StructField:
    """Structure field definition"""

    name: str
    type: str


@dataclass
class StructSchema:
    """Structure schema definition"""

    name: str
    fields: List[StructField]


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
class LogInfo:
    """Overall datalog file information"""

    filename: str
    file_size: int
    duration: float
    start_time: float
    end_time: float
    total_records: int
    signal_count: int


class DataLogManager:
    """High-level interface for WPILib datalog analysis"""

    def __init__(self):
        self.reader = None
        self.filename = None
        self.signals: Dict[str, SignalInfo] = {}
        self.signal_data: Dict[str, List[SignalValue]] = defaultdict(list)
        self.log_info: Optional[LogInfo] = None
        self._preloaded_signals: set = set()
        self.access_lock = threading.Lock()

        # Struct schema support
        self.struct_schemas: Dict[str, StructSchema] = {}

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
                self._generate_log_info()

                logger.info(f"Successfully loaded datalog: {filename}")
                return True

        except Exception as e:
            logger.error(f"Error loading datalog {filename}: {e}")
            return False

    def _parse_schema_definition(self, schema_string: str) -> List[StructField]:
        """Parse a schema definition string into fields"""
        fields = []

        # Schema format: "type1 field1;type2 field2;..."
        field_definitions = schema_string.split(";")

        for field_def in field_definitions:
            field_def = field_def.strip()
            if not field_def:
                continue

            # Split on last space to get type and name
            parts = field_def.rsplit(" ", 1)
            if len(parts) == 2:
                field_type, field_name = parts
                fields.append(StructField(field_name, field_type))

        return fields

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

                # Handle struct schema definitions
                if signal_type == "structschema" and signal_name.startswith(
                    "/.schema/struct:"
                ):
                    # This is a schema definition - we'll process it when we encounter data
                    pass

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

                    # Handle schema definitions
                    if (
                        signal_info.type == "structschema"
                        and signal_info.name.startswith("/.schema/struct:")
                    ):
                        schema_name = signal_info.name.replace("/.schema/struct:", "")
                        schema_string = self._decode_record_value(record, "string")
                        if schema_string:
                            fields = self._parse_schema_definition(schema_string)
                            self.struct_schemas[schema_name] = StructSchema(
                                schema_name, fields
                            )
                        continue

                    # Decode value based on type
                    value = self._decode_record_value(record, signal_info.type)

                    signal_value = SignalValue(
                        timestamp=timestamp, value=value, valid=True
                    )

                    self.signal_data[signal_info.name].append(signal_value)

    def _decode_record_value(self, record: DataLogRecord, signal_type: str) -> Any:
        """Decode a record value based on its type"""
        try:
            if signal_type.endswith("[]"):
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
                elif base_type.startswith("struct:"):
                    # Struct array - decode each element
                    raw_data = record.getRaw()
                    struct_name = base_type[7:]
                    return self._decode_struct_array(raw_data, struct_name)
            else:
                if signal_type == "boolean":
                    return record.getBoolean()
                elif signal_type == "int64":
                    return record.getInteger()
                elif signal_type == "float" or signal_type == "double":
                    return record.getDouble()
                elif signal_type == "string":
                    return record.getString()
                elif signal_type.startswith("struct:"):
                    # Handle non-array struct types
                    struct_name = signal_type[7:]  # Remove "struct:" prefix
                    return self._decode_struct(record, struct_name)
                else:
                    # Raw data for unknown types
                    raw_data = record.getRaw()
                    # Try to make it JSON-serializable
                    if isinstance(raw_data, bytes):
                        try:
                            # Try to decode as UTF-8 first
                            return raw_data.decode("utf-8")
                        except UnicodeDecodeError:
                            # If that fails, return as hex string
                            return raw_data.hex()
                    return raw_data
        except Exception as e:
            logger.warning(f"Error decoding record of type {signal_type}: {e}")
            return None

    def _decode_struct(self, record: DataLogRecord, struct_name: str) -> Dict[str, Any]:
        """Decode a struct from binary data"""
        if struct_name not in self.struct_schemas:
            logger.warning(f"Unknown struct type: {struct_name}")
            raw_data = record.getRaw()
            if isinstance(raw_data, bytes):
                return {"_raw_hex": raw_data.hex(), "_struct_type": struct_name}
            return {"_raw": raw_data, "_struct_type": struct_name}

        schema = self.struct_schemas[struct_name]
        raw_data = record.getRaw()

        if not isinstance(raw_data, bytes):
            return {
                "_error": "Expected bytes data for struct",
                "_struct_type": struct_name,
            }

        try:
            return self._decode_struct_from_bytes(raw_data, schema)
        except Exception as e:
            logger.warning(f"Error decoding struct {struct_name}: {e}")
            return {
                "_raw_hex": raw_data.hex(),
                "_struct_type": struct_name,
                "_error": str(e),
            }

    def _decode_struct_from_bytes(
        self, data: bytes, schema: StructSchema
    ) -> Dict[str, Any]:
        """Decode struct data from bytes using schema"""
        result = {"_struct_type": schema.name}
        offset = 0

        for field in schema.fields:
            if offset >= len(data):
                logger.warning(
                    f"Not enough data for field {field.name} in struct {schema.name}"
                )
                break

            try:
                if field.type == "double":
                    if offset + 8 <= len(data):
                        value = struct.unpack("<d", data[offset : offset + 8])[0]
                        result[field.name] = value
                        offset += 8
                    else:
                        logger.warning(
                            f"Not enough bytes for double field {field.name}"
                        )
                        break

                elif field.type == "float":
                    if offset + 4 <= len(data):
                        value = struct.unpack("<f", data[offset : offset + 4])[0]
                        result[field.name] = value
                        offset += 4
                    else:
                        logger.warning(f"Not enough bytes for float field {field.name}")
                        break

                elif field.type == "int64":
                    if offset + 8 <= len(data):
                        value = struct.unpack("<q", data[offset : offset + 8])[0]
                        result[field.name] = value
                        offset += 8
                    else:
                        logger.warning(f"Not enough bytes for int64 field {field.name}")
                        break

                elif field.type == "int32":
                    if offset + 4 <= len(data):
                        value = struct.unpack("<i", data[offset : offset + 4])[0]
                        result[field.name] = value
                        offset += 4
                    else:
                        logger.warning(f"Not enough bytes for int32 field {field.name}")
                        break

                elif field.type == "boolean":
                    if offset + 1 <= len(data):
                        value = bool(data[offset])
                        result[field.name] = value
                        offset += 1
                    else:
                        logger.warning(
                            f"Not enough bytes for boolean field {field.name}"
                        )
                        break

                elif field.type in self.struct_schemas:
                    # Nested struct
                    nested_schema = self.struct_schemas[field.type]
                    nested_size = self._calculate_struct_size(nested_schema)
                    if offset + nested_size <= len(data):
                        nested_data = data[offset : offset + nested_size]
                        nested_value = self._decode_struct_from_bytes(
                            nested_data, nested_schema
                        )
                        result[field.name] = nested_value
                        offset += nested_size
                    else:
                        logger.warning(
                            f"Not enough bytes for nested struct field {field.name}"
                        )
                        break
                else:
                    logger.warning(f"Unknown field type: {field.type}")
                    break

            except Exception as e:
                logger.warning(f"Error decoding field {field.name}: {e}")
                break

        return result

    def _calculate_struct_size(self, schema: StructSchema) -> int:
        """Calculate the size in bytes of a struct"""
        size = 0
        for field in schema.fields:
            if field.type == "double":
                size += 8
            elif field.type == "float":
                size += 4
            elif field.type == "int64":
                size += 8
            elif field.type == "int32":
                size += 4
            elif field.type == "boolean":
                size += 1
            elif field.type in self.struct_schemas:
                size += self._calculate_struct_size(self.struct_schemas[field.type])
            else:
                # Unknown type, assume 8 bytes
                size += 8
        return size

    def _decode_struct_array(
        self, data: bytes, struct_name: str
    ) -> List[Dict[str, Any]]:
        """Decode an array of structs"""
        if struct_name not in self.struct_schemas:
            return [
                {
                    "_error": f"Unknown struct type: {struct_name}",
                    "_raw_hex": data.hex(),
                }
            ]

        schema = self.struct_schemas[struct_name]
        struct_size = self._calculate_struct_size(schema)

        if len(data) % struct_size != 0:
            logger.warning(
                f"Data size {len(data)} not divisible by struct size {struct_size}"
            )

        result = []
        offset = 0

        while offset + struct_size <= len(data):
            struct_data = data[offset : offset + struct_size]
            try:
                decoded_struct = self._decode_struct_from_bytes(struct_data, schema)
                result.append(decoded_struct)
            except Exception as e:
                logger.warning(f"Error decoding struct at offset {offset}: {e}")
                break
            offset += struct_size

        return result

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

    def get_network_table_signals(self) -> List[str]:
        """Filter to only NetworkTables-sourced data"""
        nt_signals = []
        for signal_name, signal_info in self.signals.items():
            # Check metadata for NT source indicator
            if signal_info.metadata.get(
                "source"
            ) == "NetworkTables" or signal_name.startswith("NT:/"):
                nt_signals.append(signal_name)
        return nt_signals

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

    def get_struct_schemas(self) -> Dict[str, StructSchema]:
        """Get all available struct schemas"""
        return self.struct_schemas.copy()

    def get_struct_schema(self, struct_name: str) -> Optional[StructSchema]:
        """Get a specific struct schema"""
        return self.struct_schemas.get(struct_name)

    def get_struct_signals(self) -> List[str]:
        """Get all signals that use struct types"""
        struct_signals = []
        for signal_name, signal_info in self.signals.items():
            if signal_info.type.startswith("struct:"):
                struct_signals.append(signal_name)
        return struct_signals

    def get_signals_by_struct_type(self, struct_name: str) -> List[str]:
        """Get all signals that use a specific struct type"""
        target_type = f"struct:{struct_name}"
        matching_signals = []

        for signal_name, signal_info in self.signals.items():
            if signal_info.type == target_type:
                matching_signals.append(signal_name)

        return matching_signals

    def get_struct_field_as_signal(
        self, signal_name: str, field_path: str
    ) -> List[SignalValue]:
        """Extract a specific field from struct data as a virtual signal

        Args:
            signal_name: Name of the struct signal
            field_path: Dot-separated path to field (e.g., "translation.x")
        """
        if signal_name not in self.signal_data:
            return []

        field_values = []
        path_parts = field_path.split(".")

        for signal_value in self.signal_data[signal_name]:
            if not isinstance(signal_value.value, dict):
                continue

            # Navigate through the field path
            current_value = signal_value.value
            try:
                for part in path_parts:
                    if isinstance(current_value, dict) and part in current_value:
                        current_value = current_value[part]
                    else:
                        current_value = None
                        break

                if current_value is not None:
                    field_values.append(
                        SignalValue(
                            timestamp=signal_value.timestamp,
                            value=current_value,
                            valid=signal_value.valid,
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"Error extracting field {field_path} from {signal_name}: {e}"
                )

        return field_values

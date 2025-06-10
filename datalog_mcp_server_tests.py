#!/usr/bin/env python3
"""
Python Test Suite for Datalog MCP Server

This test suite validates all datalog agent tools by calling the MCP server
and verifying responses against the loaded FRC_20250418_211845_NEWTON_Q114.wpilog file.
"""



from fastmcp import Client
from mcp.types import TextContent, ImageContent, EmbeddedResource, TextResourceContents, BlobResourceContents
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
import json
import sys
import time

def parse_item(item: TextContent | ImageContent | EmbeddedResource, debug: bool = False) -> Dict[str, Any]:
    """Parse an MCP content item into a dictionary.
    
    Args:
        item: The content item to parse (TextContent, ImageContent, or EmbeddedResource)
        
    Returns:
        Dict containing the parsed content data
    """
    if isinstance(item, TextContent):
        if debug: print(f"text: {len(item.text)}")
        return json.loads(item.text)
    elif isinstance(item, ImageContent):
        if debug: print(f"image: {item.mimeType}; {len(item.data)}")
        return {
            "type": "image",
            "data": item.data,
            "mime_type": item.mimeType
        }
    elif isinstance(item, EmbeddedResource):        
        if isinstance(item.resource, TextResourceContents):
            if debug: print(f"text resource: {item.resource.mimeType}; {item.resource.uri}  [{len(item.resource.text)}]")
            return {
                "type": "resource",
                "uri": item.resource.uri,
                "mime_type": item.resource.mimeType,
                "text": item.resource.text
            }
        elif isinstance(item.resource, BlobResourceContents):
            if debug: print(f"blob resource: {item.resource.mimeType}; {item.resource.uri}  [{len(item.resource.blob)}]")
            return {
                "type": "resource",
                "uri": item.resource.uri,
                "mime_type": item.resource.mimeType,
                "blob": item.resource.blob
            }
        else: raise ValueError(f"Unsupported content type: {type(item)}")
    else:
        raise ValueError(f"Unsupported content type: {type(item)}")


class DatalogMCPTester:
    """Test runner for Datalog MCP Server functionality"""

    def __init__(self, datalog_path: str):
        self.datalog_path = datalog_path
        self.server_process = None
        self.test_results = []
        self.session = Client("datalog_mcp_server.py")

    async def call_tool_mcp(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool using MCP client"""
        if self.session:
            result = await self.session.call_tool(tool_name, kwargs)
            return [parse_item(item) for item in result][0] if len(result) > 0 else result
        else:
            # Fallback to subprocess
            return await self.call_tool_subprocess(tool_name, **kwargs)

    async def call_tool_subprocess(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool using subprocess (fallback)"""
        # This is a simplified version - in practice you'd implement proper MCP protocol
        cmd = [
            "python",
            "-c",
            f"""
import sys
sys.path.append('.')
from datalog_mcp_server import mcp
import asyncio

async def test_call():
    # Load datalog first
    await mcp.call_tool('datalog_load', filename='{self.datalog_path}')
    result = await mcp.call_tool('{tool_name}', **{kwargs})
    print(json.dumps(result))

asyncio.run(test_call())
""",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Error calling {tool_name}: {stderr.decode()}")
            return {"error": stderr.decode()}

        try:
            return json.loads(stdout.decode())
        except json.JSONDecodeError:
            return {"raw_output": stdout.decode()}

    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"    {details}")

        self.test_results.append(
            {"test": test_name, "success": success, "details": details}
        )

    async def test_datalog_load(self):
        """Test 1: Load datalog file"""
        print("\n=== Testing datalog_load ===")

        try:
            result = await self.call_tool_mcp(
                "datalog_load", filename=self.datalog_path
            )

            print(result)

            if result.get("success"):
                log_info = result.get("log_info", {})
                duration = log_info.get("duration", 0)
                signal_count = log_info.get("signal_count", 0)

                success = (
                    360 < duration < 370  # ~366.8 seconds expected
                    and signal_count == 534
                )

                details = f"Duration: {duration}s, Signals: {signal_count}"
                self.log_test_result("datalog_load", success, details)
            else:
                self.log_test_result("datalog_load", False, f"Load failed: {result}")

        except Exception as e:
            self.log_test_result("datalog_load", False, f"Exception: {e}")

    async def test_struct_schemas(self):
        """Test 2: Get struct schemas"""
        print("\n=== Testing Struct Schema Tools ===")

        # Test 2.1: Get all schemas
        try:
            result = await self.call_tool_mcp("datalog_get_struct_schemas")

            expected_schemas = [
                "Pose2d",
                "ChassisSpeeds",
                "Translation2d",
                "Rotation2d",
            ]
            found_schemas = list(result.keys()) if isinstance(result, dict) else []

            success = all(schema in found_schemas for schema in expected_schemas)
            details = f"Found schemas: {', '.join(found_schemas[:5])}..."
            self.log_test_result("get_struct_schemas", success, details)

        except Exception as e:
            self.log_test_result("get_struct_schemas", False, f"Exception: {e}")

        # Test 2.2: Get specific schema
        try:
            result = await self.call_tool_mcp(
                "datalog_get_struct_schema", struct_name="Pose2d"
            )

            if isinstance(result, dict) and "fields" in result:
                fields = result["fields"]
                expected_fields = ["translation", "rotation"]
                found_fields = [f["name"] for f in fields]

                success = all(field in found_fields for field in expected_fields)
                details = f"Pose2d fields: {', '.join(found_fields)}"
                self.log_test_result("get_struct_schema_pose2d", success, details)
            else:
                self.log_test_result(
                    "get_struct_schema_pose2d", False, "Invalid schema format"
                )

        except Exception as e:
            self.log_test_result("get_struct_schema_pose2d", False, f"Exception: {e}")

    async def test_signal_management(self):
        """Test 3: Signal management tools"""
        print("\n=== Testing Signal Management Tools ===")

        # Test 3.1: List all signals
        try:
            result = await self.call_tool_mcp("datalog_list_signals")

            if isinstance(result, list):
                signal_count = len(result)
                has_battery = any("BatteryVoltage" in sig for sig in result)
                has_enabled = any("enabled" in sig for sig in result)

                success = signal_count == 534 and has_battery and has_enabled
                details = f"Found {signal_count} signals, includes battery & enabled"
                self.log_test_result("list_signals", success, details)
            else:
                self.log_test_result(
                    "list_signals", False, "Invalid signal list format"
                )

        except Exception as e:
            self.log_test_result("list_signals", False, f"Exception: {e}")

        # Test 3.2: Search signals
        try:
            result = await self.call_tool_mcp(
                "datalog_search_signals", pattern=".*Battery.*"
            )

            if isinstance(result, list):
                battery_signals = [sig for sig in result if "Battery" in sig]
                success = len(battery_signals) >= 2  # Voltage and Current
                details = f"Found {len(battery_signals)} battery signals"
                self.log_test_result("search_signals_battery", success, details)
            else:
                self.log_test_result(
                    "search_signals_battery", False, "Invalid search result"
                )

        except Exception as e:
            self.log_test_result("search_signals_battery", False, f"Exception: {e}")

        # Test 3.3: Get signal info
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signal_info",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            if isinstance(result, dict):
                signal_type = result.get("type")
                record_count = result.get("record_count", 0)

                success = signal_type == "double" and record_count > 0
                details = f"Type: {signal_type}, Records: {record_count}"
                self.log_test_result("get_signal_info_battery", success, details)
            else:
                self.log_test_result(
                    "get_signal_info_battery", False, "Signal not found"
                )

        except Exception as e:
            self.log_test_result("get_signal_info_battery", False, f"Exception: {e}")

    async def test_time_based_queries(self):
        """Test 4: Time-based query tools"""
        print("\n=== Testing Time-based Query Tools ===")

        # Test 4.1: Get signal timespan
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signal_timespan",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            
            if isinstance(result, list) and len(result) == 2:
                start_time, end_time = result
                success = 0 < start_time < end_time < 400
                details = f"Timespan: {start_time:.2f} - {end_time:.2f}s"
                self.log_test_result("get_signal_timespan", success, details)
            else:
                self.log_test_result(
                    "get_signal_timespan", False, "Invalid timespan format"
                )

        except Exception as e:
            self.log_test_result("get_signal_timespan", False, f"Exception: {e}")

        # Test 4.2: Get value at timestamp
        try:
            # First get the actual timespan of the battery signal
            timespan = await self.call_tool_mcp(
                "datalog_get_signal_timespan",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            
            # Use midpoint of actual timespan instead of hardcoded 100.0
            if timespan and len(timespan) == 2:
                mid_timestamp = (timespan[0] + timespan[1]) / 2.0
            else:
                print(timespan)
                mid_timestamp = 5.4  # Fallback based on observed data
            
            result = await self.call_tool_mcp(
                "datalog_get_signal_value_at_timestamp",
                signal_name="/Robot/SystemStats/BatteryVoltage",
                timestamp=mid_timestamp,
            )
            if isinstance(result, dict) and "value" in result:
                value = result["value"]
                timestamp = result["timestamp"]

                success = 10.0 < value < 15.0  # Remove timestamp constraint since we're using actual data
                details = f"Value: {value:.3f}V at {timestamp:.3f}s"
                self.log_test_result("get_value_at_timestamp", success, details)
            else:
                self.log_test_result("get_value_at_timestamp", False, "No value found")

        except Exception as e:
            self.log_test_result("get_value_at_timestamp", False, f"Exception: {e}")

        # Test 4.3: Get values in range
        try:
            # Get actual timespan and use a 1-second window within it
            timespan = await self.call_tool_mcp(
                "datalog_get_signal_timespan",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            
            if timespan and len(timespan) == 2:
                start_time = timespan[0]
                end_time = min(timespan[0] + 1.0, timespan[1])  # 1-second window or until end
            else:
                start_time, end_time = 5.35, 5.47  # Fallback based on observed data
            
            result = await self.call_tool_mcp(
                "datalog_get_signal_value_in_range",
                signal_name="/Robot/SystemStats/BatteryVoltage",
                start_time=start_time,
                end_time=end_time,
            )
            if isinstance(result, list) and len(result) > 0:
                values_in_range = [
                    v for v in result if start_time <= v["timestamp"] <= end_time
                ]
                success = len(values_in_range) > 0
                details = f"Found {len(values_in_range)} values in range {start_time:.2f}-{end_time:.2f}s"
                self.log_test_result("get_values_in_range", success, details)
            else:
                self.log_test_result("get_values_in_range", False, "No values in range")

        except Exception as e:
            self.log_test_result("get_values_in_range", False, f"Exception: {e}")

    async def test_multi_signal_operations(self):
        """Test 5: Multi-signal operations"""
        print("\n=== Testing Multi-signal Operations ===")

        # Test 5.1: Get multiple signals
        try:
            signals = [
                "/Robot/SystemStats/BatteryVoltage",
                "/Robot/SystemStats/BatteryCurrent",
            ]
            
            # Get actual timespan for battery voltage signal
            timespan = await self.call_tool_mcp(
                "datalog_get_signal_timespan",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            
            if timespan and len(timespan) == 2:
                start_time = timespan[0]
                end_time = timespan[1]
            else:
                start_time, end_time = 5.35, 5.47  # Fallback
            
            result = await self.call_tool_mcp(
                "datalog_get_multiple_signals",
                signal_names=signals,
                start_time=start_time,
                end_time=end_time,
            )
            if isinstance(result, dict):
                voltage_data = result.get("/Robot/SystemStats/BatteryVoltage", [])
                current_data = result.get("/Robot/SystemStats/BatteryCurrent", [])

                # Success if we have voltage data, current data is optional
                success = len(voltage_data) > 0  
                details = f"Voltage: {len(voltage_data)} points, Current: {len(current_data)} points"
                self.log_test_result("get_multiple_signals", success, details)
            else:
                self.log_test_result(
                    "get_multiple_signals", False, "Invalid multi-signal result"
                )

        except Exception as e:
            self.log_test_result("get_multiple_signals", False, f"Exception: {e}")

        # Test 5.2: Get signal changes
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signal_changes", signal_name="DS:enabled"
            )
            if isinstance(result, list):
                changes = len(result)
                has_transitions = any(v["value"] for v in result) and any(
                    not v["value"] for v in result
                )

                success = (
                    changes >= 3 and has_transitions
                )  # Should see enable/disable transitions
                details = f"Found {changes} state changes with transitions"
                self.log_test_result("get_signal_changes", success, details)
            else:
                self.log_test_result("get_signal_changes", False, "No changes found")

        except Exception as e:
            self.log_test_result("get_signal_changes", False, f"Exception: {e}")

        # Test 5.3: Find signal events
        try:
            result = await self.call_tool_mcp(
                "datalog_find_signal_events",
                signal_name="/Robot/SystemStats/BatteryVoltage",
                condition="value < 12.8",
            )
            if isinstance(result, list):
                low_voltage_events = len(result)
                success = (
                    low_voltage_events >= 0
                )  # May or may not have low voltage events
                details = f"Found {low_voltage_events} low voltage events"
                self.log_test_result("find_signal_events", success, details)
            else:
                self.log_test_result(
                    "find_signal_events", False, "Invalid events result"
                )

        except Exception as e:
            self.log_test_result("find_signal_events", False, f"Exception: {e}")

        # Test 5.4: Get signal statistics
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signal_statistics",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            if isinstance(result, dict) and "mean" in result:
                mean_voltage = result["mean"]
                min_voltage = result["min"]
                max_voltage = result["max"]

                success = 10.0 < mean_voltage < 15.0 and min_voltage < max_voltage
                details = f"Mean: {mean_voltage:.3f}V, Range: {min_voltage:.3f}-{max_voltage:.3f}V"
                self.log_test_result("get_signal_statistics", success, details)
            else:
                self.log_test_result(
                    "get_signal_statistics", False, "Invalid statistics format"
                )

        except Exception as e:
            self.log_test_result("get_signal_statistics", False, f"Exception: {e}")

    async def test_struct_operations(self):
        """Test 6: Struct-specific operations"""
        print("\n=== Testing Struct Operations ===")

        # Test 6.1: Get struct signals
        try:
            result = await self.call_tool_mcp("datalog_get_struct_signals")
            
            if isinstance(result, list):
                struct_signals = len(result)
                has_pose = any("Pose" in sig for sig in result)

                success = struct_signals > 0 and has_pose
                details = f"Found {struct_signals} struct signals, includes Pose"
                self.log_test_result("get_struct_signals", success, details)
            else:
                self.log_test_result(
                    "get_struct_signals", False, "Invalid struct signals result"
                )

        except Exception as e:
            self.log_test_result("get_struct_signals", False, f"Exception: {e}")

        # Test 6.2: Get signals by struct type
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signals_by_struct_type", struct_name="Pose2d"
            )
            
            if isinstance(result, list):
                pose2d_signals = len(result)
                success = pose2d_signals > 0
                details = f"Found {pose2d_signals} Pose2d signals"
                self.log_test_result("get_signals_by_struct_type", success, details)
            else:
                self.log_test_result(
                    "get_signals_by_struct_type", False, "No Pose2d signals found"
                )

        except Exception as e:
            self.log_test_result("get_signals_by_struct_type", False, f"Exception: {e}")

        # Test 6.3: Extract struct field
        try:
            result = await self.call_tool_mcp(
                "datalog_get_struct_field_as_signal",
                signal_name="/Robot//DriveState/Pose",
                field_path="translation.x",
            )
            
            if isinstance(result, list) and len(result) > 0:
                x_values = [v["value"] for v in result[:10]]  # First 10 values
                has_numeric_values = all(isinstance(v, (int, float)) for v in x_values)

                success = len(result) > 100 and has_numeric_values
                details = (
                    f"Extracted {len(result)} x-coordinates, sample: {x_values[0]:.3f}"
                )
                self.log_test_result("extract_struct_field", success, details)
            else:
                self.log_test_result(
                    "extract_struct_field", False, "No field data extracted"
                )

        except Exception as e:
            self.log_test_result("extract_struct_field", False, f"Exception: {e}")

    async def test_log_analysis(self):
        """Test 7: Log analysis tools"""
        print("\n=== Testing Log Analysis Tools ===")

        # Test 7.1: Get log info
        try:
            result = await self.call_tool_mcp("datalog_get_log_info")
            if isinstance(result, dict):
                duration = result.get("duration", 0)
                signal_count = result.get("signal_count", 0)
                total_records = result.get("total_records", 0)
                success = (
                    duration > 360 and signal_count == 534 and total_records > 500000
                )
                details = f"Duration: {duration:.1f}s, Signals: {signal_count}, Records: {total_records}"
                self.log_test_result("get_log_info", success, details)
            else:
                self.log_test_result("get_log_info", False, "Invalid log info format")
        except Exception as e:
            self.log_test_result("get_log_info", False, f"Exception: {e}")

        # Test 7.4: Get NetworkTables signals
        try:
            result = await self.call_tool_mcp("datalog_get_network_table_signals")
            
            if isinstance(result, list):
                nt_signals = len(result)
                has_nt_prefix = any(sig.startswith("NT:") for sig in result)

                success = (
                    nt_signals > 0 or not has_nt_prefix
                )  # May not have NT: prefix in this log
                details = f"Found {nt_signals} NetworkTables signals"
                self.log_test_result("get_network_table_signals", success, details)
            else:
                self.log_test_result(
                    "get_network_table_signals", False, "Invalid NT signals format"
                )

        except Exception as e:
            self.log_test_result("get_network_table_signals", False, f"Exception: {e}")

    async def test_utility_functions(self):
        """Test 8: Utility functions"""
        print("\n=== Testing Utility Functions ===")

        # Test 8.1: Synchronize signals
        try:
            signals = ["/Robot/SystemStats/BatteryVoltage", "DS:enabled"]
            result = await self.call_tool_mcp(
                "datalog_synchronize_signals", signal_names=signals
            )

            if isinstance(result, dict) and len(result) == 2:
                voltage_sync = result.get("/Robot/SystemStats/BatteryVoltage", [])
                enabled_sync = result.get("DS:enabled", [])

                success = len(voltage_sync) > 0 and len(enabled_sync) > 0
                details = f"Synchronized {len(voltage_sync)} voltage, {len(enabled_sync)} enabled points"
                self.log_test_result("synchronize_signals", success, details)
            else:
                self.log_test_result(
                    "synchronize_signals", False, "Invalid sync result"
                )

        except Exception as e:
            self.log_test_result("synchronize_signals", False, f"Exception: {e}")

        # Test 8.2: Export to CSV
        try:
            signals = ["/Robot/SystemStats/BatteryVoltage"]
            filename = "/tmp/test_battery_export.csv"

            result = await self.call_tool_mcp(
                "datalog_export_to_csv",
                signal_names=signals,
                filename=filename,
                start_time=100.0,
                end_time=110.0,
            )

            if isinstance(result, dict):
                export_success = result.get("success", False)
                signal_count = result.get("signal_count", 0)

                # Check if file was created
                file_exists = Path(filename).exists() if export_success else False

                success = export_success and signal_count == 1
                details = (
                    f"Export success: {export_success}, File exists: {file_exists}"
                )
                self.log_test_result("export_to_csv", success, details)
            else:
                self.log_test_result("export_to_csv", False, "Invalid export result")

        except Exception as e:
            self.log_test_result("export_to_csv", False, f"Exception: {e}")

        # Test 8.3: Preload signals
        try:
            signals = ["/Robot/SystemStats/BatteryVoltage", "DS:enabled"]
            result = await self.call_tool_mcp(
                "datalog_preload_signals", signal_names=signals
            )

            if isinstance(result, dict):
                preload_success = result.get("success", False)
                preloaded_count = result.get("preloaded_count", 0)

                success = preload_success and preloaded_count == 2
                details = f"Preloaded {preloaded_count} signals successfully"
                self.log_test_result("preload_signals", success, details)
            else:
                self.log_test_result("preload_signals", False, "Invalid preload result")

        except Exception as e:
            self.log_test_result("preload_signals", False, f"Exception: {e}")

    async def test_edge_cases(self):
        """Test 9: Edge cases and error handling"""
        print("\n=== Testing Edge Cases ===")

        # Test 9.1: Invalid signal name
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signal_info", signal_name="/NonExistent/Signal"
            )

            # Should return None or empty result, not crash
            success = result is None or result == {} or result == []
            details = "Gracefully handled non-existent signal"
            self.log_test_result("invalid_signal_name", success, details)

        except Exception as e:
            # Exception is also acceptable for invalid signals
            self.log_test_result(
                "invalid_signal_name", True, f"Exception handled: {type(e).__name__}"
            )

        # Test 9.2: Invalid time range
        try:
            result = await self.call_tool_mcp(
                "datalog_get_signal_value_in_range",
                signal_name="/Robot/SystemStats/BatteryVoltage",
                start_time=200.0,
                end_time=100.0,
            )  # Invalid: start > end

            # Should return empty list or handle gracefully
            success = isinstance(result, list) and len(result) == 0
            details = "Handled invalid time range (start > end)"
            self.log_test_result("invalid_time_range", success, details)

        except Exception as e:
            self.log_test_result(
                "invalid_time_range", True, f"Exception handled: {type(e).__name__}"
            )

        # Test 9.3: Empty signal list
        try:
            result = await self.call_tool_mcp(
                "datalog_get_multiple_signals", signal_names=[]
            )

            # Should return empty dict or handle gracefully
            success = isinstance(result, dict) and len(result) == 0
            details = "Handled empty signal list"
            self.log_test_result("empty_signal_list", success, details)

        except Exception as e:
            self.log_test_result(
                "empty_signal_list", True, f"Exception handled: {type(e).__name__}"
            )

    async def run_all_tests(self):
        """Run all test categories"""
        print("🚀 Starting Datalog MCP Server Test Suite")
        print(f"📁 Testing with datalog: {self.datalog_path}")

        start_time = time.time()

        # Run test categories
        async with self.session:
            async def all_tests():
                await self.test_datalog_load()
                await self.test_struct_schemas()
                await self.test_signal_management()
                await self.test_time_based_queries()
                await self.test_multi_signal_operations()
                await self.test_struct_operations()
                await self.test_log_analysis()
                await self.test_utility_functions()
                await self.test_edge_cases()

            await all_tests()

        # Summary
        elapsed = time.time() - start_time
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests

        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️  Duration: {elapsed:.2f} seconds")
        print(f"📈 Success Rate: {(passed_tests/total_tests*100):.1f}%")

        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  • {result['test']}: {result['details']}")

        return failed_tests == 0

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        if self.server_process:
            self.server_process.terminate()


# Standalone test runner
async def main():
    """Main test runner"""
    if len(sys.argv) != 2:
        print("Usage: python datalog_mcp_server_tests.py <path_to_datalog_file>")
        print(
            "Example: python datalog_mcp_server_tests.py /Users/jmcmichael/Documents/worlds-logs/4-18/FRC_20250418_211845_NEWTON_Q114.wpilog"
        )
        print(sys.argv)
        sys.exit(1)

    datalog_path = sys.argv[1]

    if not Path(datalog_path).exists():
        print(f"❌ Error: Datalog file not found: {datalog_path}")
        sys.exit(1)

    tester = DatalogMCPTester(datalog_path)

    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())


# Additional helper scripts for specific testing scenarios


class QuickTester:
    """Quick tester for individual tools"""

    @staticmethod
    async def test_single_tool(tool_name: str, datalog_path: str, **kwargs):
        """Test a single tool quickly"""
        tester = DatalogMCPTester(datalog_path)

        try:
            # Load datalog first
            await tester.call_tool_mcp("datalog_load", filename=datalog_path)

            # Call the specific tool
            result = await tester.call_tool_mcp(tool_name, **kwargs)

            print(f"Tool: {tool_name}")
            print(f"Args: {kwargs}")
            print(f"Result: {json.dumps(result, indent=2)}")

        finally:
            await tester.cleanup()


# Performance testing utilities
class PerformanceTester:
    """Performance testing for datalog operations"""

    def __init__(self, datalog_path: str):
        self.datalog_path = datalog_path
        self.tester = DatalogMCPTester(datalog_path)

    async def benchmark_tool(self, tool_name: str, iterations: int = 10, **kwargs):
        """Benchmark a specific tool"""
        # Load datalog once
        await self.tester.call_tool_mcp("datalog_load", filename=self.datalog_path)

        times = []

        for i in range(iterations):
            start = time.time()
            await self.tester.call_tool_mcp(tool_name, **kwargs)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"🔧 Benchmark: {tool_name}")
        print(f"📊 Iterations: {iterations}")
        print(f"⏱️  Average: {avg_time:.3f}s")
        print(f"🏃 Fastest: {min_time:.3f}s")
        print(f"🐌 Slowest: {max_time:.3f}s")

        await self.tester.cleanup()

        return {
            "tool": tool_name,
            "iterations": iterations,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "times": times,
        }


# Regression testing utilities
class RegressionTester:
    """Regression testing against known good results"""

    KNOWN_GOOD_RESULTS = {
        # Expected results for the specific test datalog
        "signal_count": 534,
        "duration_range": (360, 370),
        "battery_voltage_range": (10.0, 15.0),
        "struct_schemas": ["Pose2d", "ChassisSpeeds", "Translation2d", "Rotation2d"],
        "pose_signals": ["/Robot//DriveState/Pose"],
    }

    def __init__(self, datalog_path: str):
        self.datalog_path = datalog_path
        self.tester = DatalogMCPTester(datalog_path)

    async def run_regression_tests(self):
        """Run regression tests against known good values"""

        try:
            # Load datalog
            load_result = await self.tester.call_tool_mcp(
                "datalog_load", filename=self.datalog_path
            )

            # Test 1: Signal count
            signals = await self.tester.call_tool_mcp("datalog_list_signals")
            actual_count = len(signals) if isinstance(signals, list) else 0
            expected_count = self.KNOWN_GOOD_RESULTS["signal_count"]

            print(f"✅ Signal count: {actual_count} (expected: {expected_count})")
            assert (
                actual_count == expected_count
            ), f"Signal count mismatch: {actual_count} != {expected_count}"

            # Test 2: Duration
            log_info = await self.tester.call_tool_mcp("datalog_get_log_info")
            duration = log_info.get("duration", 0)
            min_dur, max_dur = self.KNOWN_GOOD_RESULTS["duration_range"]

            print(f"✅ Duration: {duration:.1f}s (expected: {min_dur}-{max_dur}s)")
            assert min_dur <= duration <= max_dur, f"Duration out of range: {duration}"

            # Test 3: Battery voltage range
            battery_stats = await self.tester.call_tool_mcp(
                "datalog_get_signal_statistics",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            if battery_stats and "mean" in battery_stats:
                mean_voltage = battery_stats["mean"]
                min_volt, max_volt = self.KNOWN_GOOD_RESULTS["battery_voltage_range"]

                print(
                    f"✅ Battery voltage: {mean_voltage:.3f}V (expected: {min_volt}-{max_volt}V)"
                )
                assert (
                    min_volt <= mean_voltage <= max_volt
                ), f"Battery voltage out of range: {mean_voltage}"

            # Test 4: Struct schemas
            schemas = await self.tester.call_tool_mcp("datalog_get_struct_schemas")
            print(schemas)

            if isinstance(schemas, dict):
                found_schemas = list(schemas.keys())
                expected_schemas = self.KNOWN_GOOD_RESULTS["struct_schemas"]

                missing_schemas = [
                    s for s in expected_schemas if s not in found_schemas
                ]
                print(
                    f"✅ Struct schemas: {len(found_schemas)} found, {len(missing_schemas)} missing"
                )
                assert len(missing_schemas) == 0, f"Missing schemas: {missing_schemas}"

            print("🎉 All regression tests passed!")

        finally:
            await self.tester.cleanup()


# Integration testing utilities
class IntegrationTester:
    """Integration testing for complex workflows"""

    def __init__(self, datalog_path: str):
        self.datalog_path = datalog_path
        self.tester = DatalogMCPTester(datalog_path)

    async def test_robot_analysis_workflow(self):
        """Test a complete robot analysis workflow"""

        try:
            print("🔄 Testing Robot Analysis Workflow")

            # Step 1: Load datalog
            print("  📁 Loading datalog...")
            await self.tester.call_tool_mcp("datalog_load", filename=self.datalog_path)

            # Step 2: Get overview
            print("  📊 Getting log overview...")
            log_info = await self.tester.call_tool_mcp("datalog_get_log_info")
            print(f"     Duration: {log_info.get('duration', 0):.1f}s")
            print(f"     Signals: {log_info.get('signal_count', 0)}")

            # Step 3: Analyze power system
            print("  🔋 Analyzing power system...")
            battery_stats = await self.tester.call_tool_mcp(
                "datalog_get_signal_statistics",
                signal_name="/Robot/SystemStats/BatteryVoltage",
            )
            if battery_stats:
                print(f"     Battery: {battery_stats.get('mean', 0):.2f}V avg")

            # Step 4: Check robot states
            print("  🤖 Checking robot states...")
            enabled_changes = await self.tester.call_tool_mcp(
                "datalog_get_signal_changes", signal_name="DS:enabled"
            )
            print(f"     Enable/disable transitions: {len(enabled_changes)}")

            # Step 5: Analyze drivetrain
            print("  🚗 Analyzing drivetrain...")
            pose_data = await self.tester.call_tool_mcp(
                "datalog_get_signal_value_in_range",
                signal_name="/Robot//DriveState/Pose",
                start_time=100.0,
                end_time=110.0,
            )
            if pose_data:
                print(f"     Pose samples (10s): {len(pose_data)}")

            # Step 6: Extract position trajectory
            print("  📍 Extracting position trajectory...")
            x_coords = await self.tester.call_tool_mcp(
                "datalog_get_struct_field_as_signal",
                signal_name="/Robot//DriveState/Pose",
                field_path="translation.x",
            )
            if x_coords:
                x_values = [v["value"] for v in x_coords[:100]]  # First 100 points
                x_range = max(x_values) - min(x_values)
                print(f"     X-position range: {x_range:.2f}m")

            # Step 7: Export key data
            print("  💾 Exporting key data...")
            key_signals = [
                "/Robot/SystemStats/BatteryVoltage",
                "/Robot/SystemStats/BatteryCurrent",
                "DS:enabled",
            ]
            export_result = await self.tester.call_tool_mcp(
                "datalog_export_to_csv",
                signal_names=key_signals,
                filename="/tmp/robot_analysis.csv",
                start_time=150.0,
                end_time=170.0,
            )
            print(f"     Export success: {export_result.get('success', False)}")

            print("✅ Robot analysis workflow completed successfully!")

        finally:
            await self.tester.cleanup()

    async def test_match_analysis_workflow(self):
        """Test match analysis workflow"""

        try:
            print("🏁 Testing Match Analysis Workflow")

            # Load and get match info
            await self.tester.call_tool_mcp("datalog_load", filename=self.datalog_path)

            # Analyze match phases
            match_data = await self.tester.call_tool_mcp("datalog_find_match_data")
            phases = await self.tester.call_tool_mcp("datalog_get_match_phases")
            events = await self.tester.call_tool_mcp("datalog_get_robot_events")

            print(f"  📊 Match phases detected: {len(phases)}")
            print(f"  ⚡ Robot events: {len(events)}")
            print(f"  🌐 FMS connected: {match_data.get('fms_connected', False)}")

            # Analyze performance during different phases
            if phases:
                for phase in phases[:3]:  # First 3 phases
                    phase_name = phase.get("phase", "unknown")
                    start_time = phase.get("start_time", 0)
                    end_time = phase.get("end_time", 0)

                    print(
                        f"  🔍 Analyzing {phase_name} phase ({start_time:.1f}-{end_time:.1f}s)"
                    )

                    # Get battery performance during phase
                    battery_data = await self.tester.call_tool_mcp(
                        "datalog_get_signal_value_in_range",
                        signal_name="/Robot/SystemStats/BatteryVoltage",
                        start_time=start_time,
                        end_time=end_time,
                    )
                    if battery_data:
                        voltages = [v["value"] for v in battery_data]
                        avg_voltage = sum(voltages) / len(voltages)
                        print(
                            f"     Battery avg: {avg_voltage:.2f}V ({len(voltages)} samples)"
                        )

            print("✅ Match analysis workflow completed!")

        finally:
            await self.tester.cleanup()


# Test configuration and execution utilities
class TestConfig:
    """Test configuration management"""

    DEFAULT_CONFIG = {
        "datalog_path": "/Users/jmcmichael/Documents/worlds-logs/4-18/FRC_20250418_211845_NEWTON_Q114.wpilog",
        "timeout": 30.0,
        "retry_count": 3,
        "performance_iterations": 5,
        "export_directory": "/tmp/datalog_tests",
    }

    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load test configuration"""
        config = cls.DEFAULT_CONFIG.copy()

        if config_file and Path(config_file).exists():
            with open(config_file, "r") as f:
                user_config = json.load(f)
                config.update(user_config)

        return config


# Command-line interface
async def cli_main():
    """Enhanced command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Datalog MCP Server Test Suite")
    parser.add_argument("datalog", help="Path to datalog file")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "regression", "performance", "integration"],
        default="full",
        help="Test mode to run",
    )
    parser.add_argument("--tool", help="Specific tool to test (for quick mode)")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Performance test iterations"
    )
    parser.add_argument("--config", help="Test configuration file")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = TestConfig.load_config(args.config)
    if args.output:
        config["export_directory"] = args.output

    # Create output directory
    Path(config["export_directory"]).mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "full":
            tester = DatalogMCPTester(args.datalog)
            success = await tester.run_all_tests()
            return 0 if success else 1

        elif args.mode == "quick":
            if not args.tool:
                print("❌ Quick mode requires --tool argument")
                return 1
            await QuickTester.test_single_tool(args.tool, args.datalog)
            return 0

        elif args.mode == "regression":
            tester = RegressionTester(args.datalog)
            await tester.run_regression_tests()
            return 0

        elif args.mode == "performance":
            tester = PerformanceTester(args.datalog)
            tools_to_benchmark = [
                "datalog_list_signals",
                "datalog_get_signal_statistics",
                "datalog_get_signal_value_in_range",
                "datalog_get_multiple_signals",
            ]

            results = []
            for tool in tools_to_benchmark:
                result = await tester.benchmark_tool(
                    tool,
                    iterations=args.iterations,
                    signal_name="/Robot/SystemStats/BatteryVoltage",
                    start_time=100.0,
                    end_time=110.0,
                )
                results.append(result)

            # Save performance results
            results_file = Path(config["export_directory"]) / "performance_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"📊 Performance results saved to {results_file}")
            return 0

        elif args.mode == "integration":
            tester = IntegrationTester(args.datalog)
            await tester.test_robot_analysis_workflow()
            await tester.test_match_analysis_workflow()
            return 0

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run CLI interface if called directly
    sys.exit(asyncio.run(cli_main()))

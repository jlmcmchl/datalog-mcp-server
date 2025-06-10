#!/usr/bin/env python3
"""
Specific Tests for Individual Datalog MCP Tools

This module contains focused tests for each datalog tool with specific
validation criteria and expected behaviors.
"""

import asyncio
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple
from datalog_mcp_server_tests import DatalogMCPTester


class SpecificToolTests:
    """Specific validation tests for each datalog tool"""
    
    def __init__(self, datalog_path: str):
        self.datalog_path = datalog_path
        self.tester = DatalogMCPTester(datalog_path)
        
    async def setup(self):
        """Setup test environment"""
        await self.tester.call_tool_mcp("datalog_load", filename=self.datalog_path)

    async def cleanup(self):
        """Cleanup test environment"""
        await self.tester.cleanup()

    # =========================================================================
    # STRUCT AND SCHEMA TESTS
    # =========================================================================
    
    async def test_datalog_get_struct_schemas_detailed(self):
        """Detailed test for struct schema retrieval"""
        print("🔍 Testing datalog_get_struct_schemas (detailed)")
        
        result = await self.tester.call_tool_mcp("datalog_get_struct_schemas")
        
        # Validate structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) > 0, "Should have at least one schema"
        
        # Check for required FRC schemas
        required_schemas = ["Pose2d", "Translation2d", "Rotation2d", "ChassisSpeeds"]
        for schema_name in required_schemas:
            assert schema_name in result, f"Missing required schema: {schema_name}"
            
            schema = result[schema_name]
            assert "name" in schema, f"Schema {schema_name} missing 'name' field"
            assert "fields" in schema, f"Schema {schema_name} missing 'fields' field"
            assert isinstance(schema["fields"], list), f"Schema {schema_name} fields should be a list"
            
        # Validate Pose2d schema specifically
        pose2d = result["Pose2d"]
        pose2d_fields = {field["name"]: field["type"] for field in pose2d["fields"]}
        assert "translation" in pose2d_fields, "Pose2d missing translation field"
        assert "rotation" in pose2d_fields, "Pose2d missing rotation field"
        assert pose2d_fields["translation"] == "Translation2d", "Pose2d translation should be Translation2d type"
        assert pose2d_fields["rotation"] == "Rotation2d", "Pose2d rotation should be Rotation2d type"
        
        print(f"✅ Found {len(result)} schemas with all required FRC types")
        return True

    async def test_datalog_get_struct_schema_specific(self):
        """Test getting a specific struct schema"""
        print("🔍 Testing datalog_get_struct_schema (specific)")
        
        # Test valid schema
        result = await self.tester.call_tool_mcp("datalog_get_struct_schema", struct_name="ChassisSpeeds")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["name"] == "ChassisSpeeds", "Schema name should match request"
        
        # Validate ChassisSpeeds fields
        fields = {field["name"]: field["type"] for field in result["fields"]}
        expected_fields = {"vx": "double", "vy": "double", "omega": "double"}
        
        for field_name, field_type in expected_fields.items():
            assert field_name in fields, f"ChassisSpeeds missing field: {field_name}"
            assert fields[field_name] == field_type, f"ChassisSpeeds {field_name} should be {field_type}"
            
        # Test invalid schema
        invalid_result = await self.tester.call_tool_mcp("datalog_get_struct_schema", struct_name="NonExistentSchema")
        assert invalid_result is None or invalid_result == {} or invalid_result == [], "Invalid schema should return None or empty dict"
        
        print(f"✅ ChassisSpeeds schema validated with {len(fields)} fields")
        return True

    async def test_datalog_get_struct_signals_validation(self):
        """Test struct signal discovery"""
        print("🔍 Testing datalog_get_struct_signals (validation)")
        
        result = await self.tester.call_tool_mcp("datalog_get_struct_signals")
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Should find at least one struct signal"
        
        # Check for expected struct signals
        expected_signals = ["/Robot//DriveState/Pose", "/Robot//DriveState/Speeds"]
        found_signals = 0
        
        for expected in expected_signals:
            if expected in result:
                found_signals += 1
                
        assert found_signals > 0, f"Should find at least one expected signal from {expected_signals}"
        
        print(f"✅ Found {len(result)} struct signals, {found_signals} expected signals")
        return True

    async def test_datalog_get_struct_field_as_signal_comprehensive(self):
        """Comprehensive test for struct field extraction"""
        print("🔍 Testing datalog_get_struct_field_as_signal (comprehensive)")
        
        # Test extracting X coordinate from pose
        x_result = await self.tester.call_tool_mcp("datalog_get_struct_field_as_signal",
                                                  signal_name="/Robot//DriveState/Pose",
                                                  field_path="translation.x")
        
        assert isinstance(x_result, list), "X result should be a list"
        assert len(x_result) > 100, "Should have substantial number of X coordinates"
        
        # Validate X coordinate values
        x_values = [v["value"] for v in x_result] 
        assert all(isinstance(v, (int, float)) for v in x_values), "All X values should be numeric"

        print(len(x_result))
        
        x_range = max(x_values) - min(x_values)
        assert x_range > 0, "X coordinates should vary (robot should move)"
        
        # Test extracting Y coordinate
        y_result = await self.tester.call_tool_mcp("datalog_get_struct_field_as_signal",
                                                  signal_name="/Robot//DriveState/Pose",
                                                  field_path="translation.y")
        
        assert len(y_result) == len(x_result), "X and Y should have same number of samples"
        
        # Test extracting rotation
        rotation_result = await self.tester.call_tool_mcp("datalog_get_struct_field_as_signal",
                                                         signal_name="/Robot//DriveState/Pose",
                                                         field_path="rotation.value")
        
        assert isinstance(rotation_result, list), "Rotation result should be a list"
        rotation_values = [v["value"] for v in rotation_result[:100]]
        assert all(-math.pi <= v <= math.pi for v in rotation_values), "Rotation values should be in valid range"
        
        # Test invalid field path
        try:
            invalid_result = await self.tester.call_tool_mcp("datalog_get_struct_field_as_signal",
                                                           signal_name="/Robot//DriveState/Pose",
                                                           field_path="nonexistent.field")
            assert len(invalid_result) == 0, "Invalid field path should return empty result"
        except Exception:
            pass  # Exception is also acceptable for invalid field path
            
        print(f"✅ Extracted {len(x_result)} pose coordinates with range {x_range:.2f}m")
        return True

    # =========================================================================
    # SIGNAL MANAGEMENT TESTS
    # =========================================================================
    
    async def test_datalog_list_signals_comprehensive(self):
        """Comprehensive signal listing test"""
        print("🔍 Testing datalog_list_signals (comprehensive)")
        
        # Test all signals
        all_signals = await self.tester.call_tool_mcp("datalog_list_signals")
        
        assert isinstance(all_signals, list), "Result should be a list"
        assert len(all_signals) == 534, f"Expected 534 signals, got {len(all_signals)}"
        
        # Check for required signal categories
        categories = {
            "system_stats": [s for s in all_signals if "SystemStats" in s],
            "driver_station": [s for s in all_signals if "DS:" in s],
            "robot_subsystems": [s for s in all_signals if "/Robot/" in s],
            "network_tables": [s for s in all_signals if "NT:" in s],
        }
        
        assert len(categories["system_stats"]) > 0, "Should have system stats signals"
        assert len(categories["driver_station"]) > 0, "Should have driver station signals"
        assert len(categories["robot_subsystems"]) > 0, "Should have robot subsystem signals"
        
        # Test with pattern
        battery_signals = await self.tester.call_tool_mcp("datalog_list_signals", pattern=".*Battery.*")
        assert isinstance(battery_signals, list), "Pattern result should be a list"
        assert len(battery_signals) >= 2, "Should find at least battery voltage and current"
        
        print(f"✅ Found {len(all_signals)} total signals across {len(categories)} categories")
        return True

    async def test_datalog_get_signal_info_detailed(self):
        """Detailed signal info validation"""
        print("🔍 Testing datalog_get_signal_info (detailed)")
        
        # Test battery voltage signal
        battery_info = await self.tester.call_tool_mcp("datalog_get_signal_info",
                                                      signal_name="/Robot/SystemStats/BatteryVoltage")
        
        assert isinstance(battery_info, dict), "Signal info should be a dictionary"
        
        required_fields = ["name", "type", "first_timestamp", "last_timestamp", "record_count", "entry_id"]
        for field in required_fields:
            assert field in battery_info, f"Signal info missing field: {field}"
            
        assert battery_info["type"] == "double", "Battery voltage should be double type"
        assert battery_info["record_count"] > 0, "Should have some records"
        assert 0 < battery_info["first_timestamp"] < battery_info["last_timestamp"], "Valid timestamp range"
        
        # Test boolean signal
        enabled_info = await self.tester.call_tool_mcp("datalog_get_signal_info", signal_name="DS:enabled")
        assert enabled_info["type"] == "boolean", "Robot enabled should be boolean type"
        
        # Test struct signal
        pose_info = await self.tester.call_tool_mcp("datalog_get_signal_info",
                                                   signal_name="/Robot//DriveState/Pose")
        assert pose_info["type"].startswith("struct:"), "Pose should be struct type"
        
        print(f"✅ Validated signal info for {battery_info['name']} with {battery_info['record_count']} records")
        return True

    # =========================================================================
    # TIME-BASED QUERY TESTS
    # =========================================================================
    
    async def test_datalog_get_signal_value_at_timestamp_precision(self):
        """Test timestamp precision for value retrieval"""
        print("🔍 Testing datalog_get_signal_value_at_timestamp (precision)")
        
        # Get signal timespan first
        timespan = await self.tester.call_tool_mcp("datalog_get_signal_timespan",
                                                  signal_name="/Robot/SystemStats/BatteryVoltage")
        start_time, end_time = timespan
        
        # Test at various timestamps
        test_timestamps = [
            start_time + 1.0,
            (start_time + end_time) / 2,  # Middle
            end_time - 1.0,
            start_time - 1.0,  # Before range
            end_time + 1.0,    # After range
        ]
        
        valid_results = 0
        for timestamp in test_timestamps:
            result = await self.tester.call_tool_mcp("datalog_get_signal_value_at_timestamp",
                                                    signal_name="/Robot/SystemStats/BatteryVoltage",
                                                    timestamp=timestamp)
            
            if result and "value" in result:
                valid_results += 1
                assert isinstance(result["value"], (int, float)), "Value should be numeric"
                assert 10.0 <= result["value"] <= 15.0, f"Battery voltage {result['value']} out of expected range"
                
                # Check timestamp proximity
                time_diff = abs(result["timestamp"] - timestamp)
                assert time_diff < 10.0, f"Retrieved timestamp too far from requested: {time_diff}s"
                
        assert valid_results >= 3, f"Should get valid results for most timestamps, got {valid_results}/5"
        
        print(f"✅ Retrieved valid values for {valid_results}/5 test timestamps")
        return True

    async def test_datalog_get_signal_value_in_range_boundaries(self):
        """Test range query boundary conditions"""
        print("🔍 Testing datalog_get_signal_value_in_range (boundaries)")
        
        # Test normal range
        normal_range = await self.tester.call_tool_mcp("datalog_get_signal_value_in_range",
                                                      signal_name="/Robot/SystemStats/BatteryVoltage",
                                                      start_time=0.0, end_time=1000.0)
        
        assert isinstance(normal_range, list), "Range result should be a list"
        assert len(normal_range) > 0, "Should find values in normal range"

        print(normal_range)
        
        # Verify all timestamps are within range
        for value in normal_range:
            assert 100.0 <= value["timestamp"] <= 110.0, f"Timestamp {value['timestamp']} outside range"
            
        # Test edge cases
        edge_cases = [
            (200.0, 200.0),    # Zero-width range
            (150.0, 149.0),    # Invalid range (start > end)
            (-10.0, -5.0),     # Before log start
            (500.0, 600.0),    # After log end
        ]
        
        for start, end in edge_cases:
            edge_result = await self.tester.call_tool_mcp("datalog_get_signal_value_in_range",
                                                         signal_name="/Robot/SystemStats/BatteryVoltage",
                                                         start_time=start, end_time=end)
            assert isinstance(edge_result, list), "Edge case should return list (possibly empty)"
            
        print(f"✅ Normal range returned {len(normal_range)} values, edge cases handled")
        return True

    # =========================================================================
    # ANALYSIS AND STATISTICS TESTS
    # =========================================================================
    
    async def test_datalog_get_signal_statistics_accuracy(self):
        """Test statistical accuracy"""
        print("🔍 Testing datalog_get_signal_statistics (accuracy)")
        
        # Get statistics for battery voltage
        stats = await self.tester.call_tool_mcp("datalog_get_signal_statistics",
                                               signal_name="/Robot/SystemStats/BatteryVoltage")
        
        assert isinstance(stats, dict), "Statistics should be a dictionary"
        
        required_stats = ["count", "min", "max", "mean", "median", "stdev"]
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
            assert isinstance(stats[stat], (int, float)), f"Statistic {stat} should be numeric"
            
        # Validate statistical relationships
        assert stats["min"] <= stats["median"] <= stats["max"], "Min <= median <= max"
        assert stats["min"] <= stats["mean"] <= stats["max"], "Min <= mean <= max"
        assert stats["stdev"] >= 0, "Standard deviation should be non-negative"
        assert stats["count"] > 0, "Should have counted some values"
        
        # Validate reasonable battery voltage ranges
        assert 10.0 <= stats["min"] <= 15.0, f"Battery min voltage {stats['min']} unrealistic"
        assert 10.0 <= stats["max"] <= 15.0, f"Battery max voltage {stats['max']} unrealistic"
        assert stats["stdev"] < 2.0, f"Battery voltage std dev {stats['stdev']} too high"
        
        # Get raw data and verify statistics manually (sample)
        sample_data = await self.tester.call_tool_mcp("datalog_get_signal_value_in_range",
                                                     signal_name="/Robot/SystemStats/BatteryVoltage",
                                                     start_time=100.0, end_time=110.0)
        
        if len(sample_data) > 10:
            sample_values = [v["value"] for v in sample_data]
            manual_mean = sum(sample_values) / len(sample_values)
            manual_min = min(sample_values)
            manual_max = max(sample_values)
            
            # Statistics should be reasonable compared to sample
            assert abs(manual_mean - stats["mean"]) < 1.0, "Sample mean should be close to overall mean"
            assert manual_min >= stats["min"], "Sample min should be >= overall min"
            assert manual_max <= stats["max"], "Sample max should be <= overall max"
            
        print(f"✅ Statistics validated: mean={stats['mean']:.3f}V, stdev={stats['stdev']:.4f}V")
        return True

    async def test_datalog_find_signal_events_conditions(self):
        """Test event finding with various conditions"""
        print("🔍 Testing datalog_find_signal_events (conditions)")
        
        test_conditions = [
            ("value < 12.0", "low voltage events"),
            ("value > 13.0", "high voltage events"),
            ("value > 12.5 and value < 13.0", "normal voltage range"),
        ]
        
        for condition, description in test_conditions:
            try:
                events = await self.tester.call_tool_mcp("datalog_find_signal_events",
                                                        signal_name="/Robot/SystemStats/BatteryVoltage",
                                                        condition=condition)
                
                assert isinstance(events, list), f"Events for '{condition}' should be a list"
                
                # Verify events meet condition
                for event in events[:10]:  # Check first 10 events
                    value = event["value"]
                    # Simple validation - in production would use safer eval
                    if "< 12.0" in condition:
                        assert value < 12.0, f"Event value {value} doesn't meet condition {condition}"
                    elif "> 13.0" in condition:
                        assert value > 13.0, f"Event value {value} doesn't meet condition {condition}"
                        
                print(f"  ✓ {description}: {len(events)} events found")
                
            except Exception as e:
                print(f"  ! {description}: Error - {e}")
                
        print("✅ Event condition testing completed")
        return True

    # =========================================================================
    # EXPORT AND UTILITY TESTS
    # =========================================================================
    
    async def test_datalog_export_to_csv_validation(self):
        """Test CSV export with validation"""
        print("🔍 Testing datalog_export_to_csv (validation)")
        
        import tempfile
        import csv
        from pathlib import Path
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Export test data
            signals = ["/Robot/SystemStats/BatteryVoltage", "/Robot/SystemStats/BatteryCurrent"]
            result = await self.tester.call_tool_mcp("datalog_export_to_csv",
                                                    signal_names=signals,
                                                    filename=temp_path,
                                                    start_time=100.0, end_time=105.0)
            
            assert result.get("success"), "Export should succeed"
            assert result.get("signal_count") == 2, "Should export 2 signals"
            
            # Validate CSV file
            assert Path(temp_path).exists(), "CSV file should be created"
            
            with open(temp_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                
                # Validate header
                assert "timestamp" in header, "CSV should have timestamp column"
                for signal in signals:
                    assert signal in header, f"CSV should have column for {signal}"
                    
                # Count data rows
                data_rows = list(reader)
                assert len(data_rows) > 0, "CSV should have data rows"
                
                # Validate first few rows
                for i, row in enumerate(data_rows[:5]):
                    assert len(row) == len(header), f"Row {i} should have {len(header)} columns"
                    
                    # Validate timestamp
                    timestamp = float(row[0])
                    assert 100.0 <= timestamp <= 105.0, f"Timestamp {timestamp} outside export range"
                    
            print(f"✅ CSV export validated: {len(data_rows)} rows, {len(header)} columns")
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            
        return True

    async def test_datalog_synchronize_signals_alignment(self):
        """Test signal synchronization alignment"""
        print("🔍 Testing datalog_synchronize_signals (alignment)")
        
        # Synchronize signals with different sampling rates
        signals = ["/Robot/SystemStats/BatteryVoltage", "DS:enabled"]
        result = await self.tester.call_tool_mcp("datalog_synchronize_signals", signal_names=signals)
        
        assert isinstance(result, dict), "Sync result should be a dictionary"
        assert len(result) == 2, "Should sync exactly 2 signals"
        
        voltage_sync = result.get("/Robot/SystemStats/BatteryVoltage", [])
        enabled_sync = result.get("DS:enabled", [])
        
        assert len(voltage_sync) > 0, "Should have synchronized voltage data"
        assert len(enabled_sync) > 0, "Should have synchronized enabled data"
        
        # Check timestamp alignment
        if len(voltage_sync) > 0 and len(enabled_sync) > 0:
            # Get common length (might be different due to signal availability)
            min_length = min(len(voltage_sync), len(enabled_sync))
            
            if min_length > 10:
                # Check that timestamps are reasonably aligned for first 10 samples
                for i in range(10):
                    if i < len(voltage_sync) and i < len(enabled_sync):
                        v_time = voltage_sync[i]["timestamp"]
                        e_time = enabled_sync[i]["timestamp"]
                        time_diff = abs(v_time - e_time)
                        
                        # Allow some tolerance for different sampling rates
                        assert time_diff < 1.0, f"Timestamps too different at index {i}: {time_diff}s"
                        
        print(f"✅ Synchronized {len(voltage_sync)} voltage and {len(enabled_sync)} enabled samples")
        return True

    # =========================================================================
    # ERROR HANDLING AND EDGE CASE TESTS
    # =========================================================================
    
    async def test_error_handling_comprehensive(self):
        """Comprehensive error handling test"""
        print("🔍 Testing error handling (comprehensive)")
        
        error_test_cases = [
            # Invalid signal names
            ("datalog_get_signal_info", {"signal_name": "/Invalid/Signal/Name"}, "invalid signal"),
            ("datalog_get_signal_statistics", {"signal_name": "/NonExistent/Signal"}, "non-existent signal"),
            
            # Invalid struct operations
            ("datalog_get_struct_schema", {"struct_name": "NonExistentStruct"}, "invalid struct"),
            ("datalog_get_struct_field_as_signal", {
                "signal_name": "/Robot//DriveState/Pose", 
                "field_path": "invalid.field.path"
            }, "invalid field path"),
            
            # Invalid time ranges
            ("datalog_get_signal_value_in_range", {
                "signal_name": "/Robot/SystemStats/BatteryVoltage",
                "start_time": 1000.0, "end_time": 900.0  # start > end
            }, "invalid time range"),
            
            # Empty parameters
            ("datalog_get_multiple_signals", {"signal_names": []}, "empty signal list"),
            ("datalog_synchronize_signals", {"signal_names": []}, "empty sync list"),
        ]
        
        handled_errors = 0
        for tool_name, params, description in error_test_cases:
            try:
                result = await self.tester.call_tool_mcp(tool_name, **params)
                
                # Result should be None, empty, or contain error information
                # Should not crash or return invalid data
                if result is None or result == {} or (isinstance(result, list) and len(result) == 0):
                    handled_errors += 1
                    print(f"  ✓ {description}: handled gracefully")
                elif isinstance(result, dict) and "error" in result:
                    handled_errors += 1
                    print(f"  ✓ {description}: returned error info")
                else:
                    print(f"  ? {description}: unexpected result type")
                    
            except Exception as e:
                # Exceptions are also acceptable for error cases
                handled_errors += 1
                print(f"  ✓ {description}: raised {type(e).__name__}")
                
        success_rate = handled_errors / len(error_test_cases)
        assert success_rate >= 0.8, f"Should handle most errors gracefully: {success_rate:.1%}"
        
        print(f"✅ Error handling: {handled_errors}/{len(error_test_cases)} cases handled properly")
        return True

    # =========================================================================
    # PERFORMANCE AND STRESS TESTS
    # =========================================================================
    
    async def test_performance_large_queries(self):
        """Test performance with large data queries"""
        print("🔍 Testing performance (large queries)")
        
        import time
        
        # Test large time range query
        start_time = time.time()
        large_range = await self.tester.call_tool_mcp("datalog_get_signal_value_in_range",
                                                     signal_name="/Robot//DriveState/Pose",
                                                     start_time=0.0, end_time=400.0)
        query_time = time.time() - start_time
        
        assert isinstance(large_range, list), "Large range should return list"
        data_points = len(large_range)
        
        # Performance expectations
        assert query_time < 10.0, f"Large query took too long: {query_time:.2f}s"
        assert data_points > 1000, f"Should get substantial data: {data_points} points"
        
        data_rate = data_points / query_time if query_time > 0 else 0
        
        # Test multiple signals query
        start_time = time.time()
        multi_signals = [
            "/Robot/SystemStats/BatteryVoltage",
            "/Robot/SystemStats/BatteryCurrent",
            "/Robot//DriveState/Pose",
            "DS:enabled"
        ]
        multi_result = await self.tester.call_tool_mcp("datalog_get_multiple_signals",
                                                      signal_names=multi_signals,
                                                      start_time=100.0, end_time=200.0)
        multi_time = time.time() - start_time
        
        assert isinstance(multi_result, dict), "Multi-signal should return dict"
        assert len(multi_result) == len(multi_signals), "Should return all requested signals"
        
        total_points = sum(len(data) for data in multi_result.values())
        multi_rate = total_points / multi_time if multi_time > 0 else 0
        
        print(f"✅ Performance: {data_points} points in {query_time:.2f}s ({data_rate:.0f} pts/s)")
        print(f"✅ Multi-signal: {total_points} points in {multi_time:.2f}s ({multi_rate:.0f} pts/s)")
        
        return True


# =========================================================================
# TEST RUNNER FOR SPECIFIC TESTS
# =========================================================================

async def run_specific_tests(datalog_path: str, test_names: Optional[List[str]] = None):
    """Run specific detailed tests"""
    
    tester = SpecificToolTests(datalog_path)
    async with tester.tester.session:
        await tester.setup()
    
        # All available tests
        all_tests = [
            ("struct_schemas_detailed", tester.test_datalog_get_struct_schemas_detailed),
            ("struct_schema_specific", tester.test_datalog_get_struct_schema_specific),
            ("struct_signals_validation", tester.test_datalog_get_struct_signals_validation),
            ("struct_field_extraction", tester.test_datalog_get_struct_field_as_signal_comprehensive),
            ("signal_listing", tester.test_datalog_list_signals_comprehensive),
            ("signal_info_detailed", tester.test_datalog_get_signal_info_detailed),
            ("timestamp_precision", tester.test_datalog_get_signal_value_at_timestamp_precision),
            ("range_boundaries", tester.test_datalog_get_signal_value_in_range_boundaries),
            ("statistics_accuracy", tester.test_datalog_get_signal_statistics_accuracy),
            ("event_conditions", tester.test_datalog_find_signal_events_conditions),
            ("csv_export_validation", tester.test_datalog_export_to_csv_validation),
            ("signal_synchronization", tester.test_datalog_synchronize_signals_alignment),
            ("error_handling", tester.test_error_handling_comprehensive),
            ("performance_large", tester.test_performance_large_queries),
        ]
        
        # Filter tests if specific names provided
        if test_names:
            tests_to_run = [(name, func) for name, func in all_tests if name in test_names]
        else:
            tests_to_run = all_tests
            
        print(f"🚀 Running {len(tests_to_run)} specific tests")
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests_to_run:
            try:
                print(f"\n--- {test_name} ---")
                success = await test_func()
                if success:
                    passed += 1
                else:
                    failed += 1
                    print(f"❌ {test_name} failed")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} failed with exception: {e}")
                
        await tester.cleanup()
    
    print(f"\n{'='*60}")
    print(f"SPECIFIC TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python specific_tests.py <datalog_path> [test_name1] [test_name2] ...")
        sys.exit(1)
        
    datalog_path = sys.argv[1]
    test_names = sys.argv[2:] if len(sys.argv) > 2 else None
    
    success = asyncio.run(run_specific_tests(datalog_path, test_names))
    sys.exit(0 if success else 1)
"""Tests for BackgroundTaskManager with performance and load testing."""

import asyncio
import logging
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from agentflow.utils.background_task_manager import BackgroundTaskManager


class TestBackgroundTaskManager:
    """Test suite for BackgroundTaskManager including performance and load tests."""

    @pytest.fixture
    def task_manager(self):
        """Create a fresh BackgroundTaskManager instance for each test."""
        return BackgroundTaskManager()

    @pytest.mark.asyncio
    async def test_create_task_basic(self, task_manager):
        """Test basic task creation and completion."""
        result = []
        
        async def simple_task():
            await asyncio.sleep(0.1)
            result.append("completed")
        
        task_manager.create_task(simple_task())
        
        # Wait for task to complete
        await task_manager.wait_for_all()
        
        assert result == ["completed"]
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_create_multiple_tasks(self, task_manager):
        """Test creating multiple tasks concurrently."""
        results = []
        
        async def numbered_task(number):
            await asyncio.sleep(0.1)
            results.append(number)
        
        # Create 5 concurrent tasks
        for i in range(5):
            task_manager.create_task(numbered_task(i))
        
        await task_manager.wait_for_all()
        
        assert len(results) == 5
        assert sorted(results) == list(range(5))
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_error_handling(self, task_manager):
        """Test that task errors are logged but don't crash the manager."""
        with patch('agentflow.utils.background_task_manager.logger') as mock_logger:

            async def failing_task():
                await asyncio.sleep(0.05)
                raise ValueError("Test error")

            task_manager.create_task(failing_task())
            
            # wait_for_all() should raise the exception
            with pytest.raises(ValueError, match="Test error"):
                await task_manager.wait_for_all()

            # Verify error was logged during task completion
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Background task raised an exception" in error_call
            assert "Test error" in str(error_call)

    @pytest.mark.asyncio
    async def test_task_cleanup_on_completion(self, task_manager):
        """Test that completed tasks are properly removed from tracking."""
        async def quick_task():
            await asyncio.sleep(0.05)
        
        # Start task
        task_manager.create_task(quick_task())
        
        # Initially should have one task
        assert len(task_manager._tasks) == 1
        
        # Wait for completion
        await asyncio.sleep(0.1)
        
        # Task should be cleaned up
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_cleanup_on_error(self, task_manager):
        """Test that failed tasks are properly removed from tracking."""
        with patch('agentflow.utils.background_task_manager.logger'):
            async def failing_task():
                await asyncio.sleep(0.05)
                raise RuntimeError("Test failure")
            
            task_manager.create_task(failing_task())
            
            # Initially should have one task
            assert len(task_manager._tasks) == 1
            
            # Wait for completion/failure
            await asyncio.sleep(0.1)
            
            # Task should be cleaned up even after failure
            assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_wait_for_all_empty(self, task_manager):
        """Test wait_for_all when no tasks are running."""
        # Should complete immediately without error
        await task_manager.wait_for_all()
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_wait_for_all_with_mixed_success_failure(self, task_manager):
        """Test wait_for_all with both successful and failing tasks."""
        results = []

        async def success_task(value):
            await asyncio.sleep(0.1)
            results.append(value)

        async def fail_task():
            await asyncio.sleep(0.05)
            raise RuntimeError("Intentional failure")

        with patch('agentflow.utils.background_task_manager.logger'):
            # Mix of success and failure tasks
            task_manager.create_task(success_task("success1"))
            task_manager.create_task(fail_task())
            task_manager.create_task(success_task("success2"))

            # wait_for_all should raise the first exception encountered
            with pytest.raises(RuntimeError, match="Intentional failure"):
                await task_manager.wait_for_all()

            # The successful tasks may not complete due to cancellation
            # This is expected behavior with asyncio.gather when one task fails

    @pytest.mark.asyncio
    async def test_wait_for_all_leaves_no_residue(self, task_manager):
        """wait_for_all() must return with the bookkeeping already cleaned up.

        ``asyncio.gather`` resolves the moment the last task finishes, but each task's
        ``add_done_callback`` cleanup is queued with ``call_soon`` and has not run yet.
        Returning at that point leaves every finished task in ``_tasks`` and
        ``_task_metadata``, so a caller reading ``pending_count`` immediately after
        sees a stale non-zero value. The burst-then-pause shape is what exposes it:
        earlier bursts get their callbacks drained by the pause, the final burst does
        not, so exactly one burst's worth is left behind.
        """
        for _ in range(5):
            for _ in range(20):

                async def job():
                    await asyncio.sleep(0.001)

                task_manager.create_task(job())
            await asyncio.sleep(0.01)

        await task_manager.wait_for_all()

        assert task_manager.pending_count == 0
        assert task_manager.get_task_count() == 0
        assert task_manager._task_metadata == {}

    @pytest.mark.asyncio
    async def test_wait_for_all_timeout_leaves_no_residue(self, task_manager):
        """The timeout path waits for real work, so it must not report residue either."""
        for _ in range(5):
            for _ in range(20):

                async def job():
                    await asyncio.sleep(0.001)

                task_manager.create_task(job())
            await asyncio.sleep(0.01)

        await task_manager.wait_for_all(timeout=15)

        assert task_manager.pending_count == 0
        assert task_manager._task_metadata == {}

    @pytest.mark.asyncio
    async def test_shutdown_reports_accurate_completion_counts(self, task_manager):
        """shutdown() derives completed_tasks from len(_tasks) after gather.

        ``cancel_all`` already awaits, which incidentally drains the done-callbacks,
        so this passes either way today. It is here to pin the reported statistics so
        a future change to that scheduling cannot silently make them lie.
        """

        async def job():
            await asyncio.sleep(0.001)

        for _ in range(10):
            task_manager.create_task(job())

        stats = await task_manager.shutdown()

        assert stats["status"] == "completed"
        assert stats["initial_tasks"] == 10
        assert stats["completed_tasks"] == 10
        assert stats["remaining_tasks"] == 0

    @pytest.mark.asyncio
    async def test_high_load_concurrent_tasks(self, task_manager):
        """Test BackgroundTaskManager under high load with many concurrent tasks."""
        num_tasks = 100
        results = []
        
        async def load_task(task_id):
            # Simulate some work
            await asyncio.sleep(0.01)
            results.append(task_id)
        
        start_time = time.time()
        
        # Create many tasks quickly
        for i in range(num_tasks):
            task_manager.create_task(load_task(i))
        
        # Wait for all to complete
        await task_manager.wait_for_all()
        
        end_time = time.time()
        
        # Verify all tasks completed
        assert len(results) == num_tasks
        assert sorted(results) == list(range(num_tasks))
        assert len(task_manager._tasks) == 0
        
        # Should complete in reasonable time (concurrent execution)
        # With 100 tasks of 0.01s each, sequential would take 1s+
        # Concurrent should be much faster
        assert end_time - start_time < 0.5

    @pytest.mark.asyncio
    async def test_memory_efficiency_under_load(self, task_manager):
        """Test memory efficiency by creating and completing many tasks in batches."""
        total_completed = 0
        
        async def memory_task(batch_id, task_id):
            await asyncio.sleep(0.001)  # Very short task
            return f"batch_{batch_id}_task_{task_id}"
        
        # Process tasks in batches to test memory management
        batch_size = 50
        num_batches = 10
        
        for batch in range(num_batches):
            # Create a batch of tasks
            for task_id in range(batch_size):
                task_manager.create_task(memory_task(batch, task_id))
            
            # Wait for this batch to complete
            await task_manager.wait_for_all()
            total_completed += batch_size
            
            # Tasks should be cleaned up after each batch
            assert len(task_manager._tasks) == 0
        
        assert total_completed == batch_size * num_batches

    @pytest.mark.asyncio
    async def test_rapid_task_creation_and_completion(self, task_manager):
        """Test rapid creation and completion of tasks."""
        completion_count = 0
        
        async def rapid_task():
            nonlocal completion_count
            completion_count += 1
        
        # Rapidly create and let tasks complete
        for _ in range(50):
            task_manager.create_task(rapid_task())
            await asyncio.sleep(0.001)  # Small delay between creations
        
        await task_manager.wait_for_all()
        
        assert completion_count == 50
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_error_rate_under_load(self, task_manager):
        """Test system stability when many tasks fail."""
        success_count = 0
        error_count = 0
        
        async def mixed_task(task_id):
            nonlocal success_count, error_count
            await asyncio.sleep(0.01)
            if task_id % 3 == 0:  # Every 3rd task fails
                error_count += 1
                raise ValueError(f"Task {task_id} failed")
            else:
                success_count += 1
        
        with patch('agentflow.utils.background_task_manager.logger'):
            # Create 60 tasks (20 will fail, 40 will succeed)
            for i in range(60):
                task_manager.create_task(mixed_task(i))
            
            # wait_for_all should raise when first task fails (task 0)
            with pytest.raises(ValueError, match="Task 0 failed"):
                await task_manager.wait_for_all()
            
            # At least the first failing task should have incremented error_count
            assert error_count >= 1

    @pytest.mark.asyncio
    async def test_long_running_vs_short_tasks(self, task_manager):
        """Test mixing long-running and short tasks."""
        short_completed = 0
        long_completed = 0
        
        async def short_task():
            nonlocal short_completed
            await asyncio.sleep(0.01)
            short_completed += 1
        
        async def long_task():
            nonlocal long_completed
            await asyncio.sleep(0.1)
            long_completed += 1
        
        # Mix of short and long tasks
        for _ in range(10):
            task_manager.create_task(short_task())
        for _ in range(3):
            task_manager.create_task(long_task())
        
        await task_manager.wait_for_all()
        
        assert short_completed == 10
        assert long_completed == 3
        assert len(task_manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_done_callback_thread_safety(self, task_manager):
        """Test that the done callback is thread-safe with concurrent task completions."""
        completion_order = []
        
        async def callback_test_task(task_id):
            await asyncio.sleep(0.01 * (task_id % 3))  # Variable delay
            completion_order.append(task_id)
        
        # Create tasks with different completion times
        for i in range(20):
            task_manager.create_task(callback_test_task(i))
        
        await task_manager.wait_for_all()
        
        # All tasks should complete
        assert len(completion_order) == 20
        assert len(task_manager._tasks) == 0
        # Order might vary due to different delays, but all should be present
        assert sorted(completion_order) == list(range(20))

    @pytest.mark.asyncio
    async def test_performance_with_cpu_bound_simulation(self, task_manager):
        """Test performance with CPU-intensive simulation."""
        results = []
        
        async def cpu_intensive_task(n):
            # Simulate CPU work with async yield points
            total = 0
            for i in range(n):
                total += i
                if i % 1000 == 0:
                    await asyncio.sleep(0)  # Yield control
            results.append(total)
        
        start_time = time.time()
        
        # Create tasks with different workloads
        for i in range(10):
            workload = 5000 + (i * 1000)  # Varying workloads
            task_manager.create_task(cpu_intensive_task(workload))
        
        await task_manager.wait_for_all()
        end_time = time.time()
        
        assert len(results) == 10
        assert len(task_manager._tasks) == 0
        # Should complete in reasonable time
        assert end_time - start_time < 2.0

    @pytest.mark.asyncio
    async def test_exception_types_handling(self, task_manager):
        """Test handling of different exception types."""
        with patch('agentflow.utils.background_task_manager.logger') as mock_logger:
            
            async def value_error_task():
                raise ValueError("Value error")
            
            async def runtime_error_task():
                raise RuntimeError("Runtime error")
            
            async def custom_error_task():
                class CustomError(Exception):
                    pass
                raise CustomError("Custom error")
            
            task_manager.create_task(value_error_task())
            task_manager.create_task(runtime_error_task())
            task_manager.create_task(custom_error_task())
            
            # wait_for_all should raise the first exception (ValueError)
            with pytest.raises(ValueError, match="Value error"):
                await task_manager.wait_for_all()
            
            # At least one error should be logged
            assert mock_logger.error.call_count >= 1

    @pytest.mark.asyncio
    async def test_stress_test_rapid_create_destroy(self, task_manager):
        """Stress test with rapid task creation and destruction."""
        total_created = 0
        total_completed = 0
        
        async def stress_task():
            nonlocal total_completed
            await asyncio.sleep(0.001)
            total_completed += 1
        
        # Rapidly create tasks in bursts
        for burst in range(5):
            for _ in range(20):
                task_manager.create_task(stress_task())
                total_created += 1
            
            # Let some tasks complete
            await asyncio.sleep(0.01)
        
        # Wait for all remaining tasks
        await task_manager.wait_for_all()
        
        assert total_created == 100
        assert total_completed == 100
        assert len(task_manager._tasks) == 0

    def test_initialization(self):
        """Test BackgroundTaskManager initialization."""
        manager = BackgroundTaskManager()
        assert isinstance(manager._tasks, set)
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_logging_configuration(self, task_manager):
        """Test that logging works correctly under load."""
        with patch('agentflow.utils.background_task_manager.logger') as mock_logger:
            
            async def logging_task():
                await asyncio.sleep(0.01)
                raise Exception("Test logging")
            
            # Create multiple failing tasks
            for _ in range(5):
                task_manager.create_task(logging_task())
            
            # wait_for_all should raise the first exception
            with pytest.raises(Exception, match="Test logging"):
                await task_manager.wait_for_all()
            
            # At least one error should be logged
            assert mock_logger.error.call_count >= 1
            
            # Check log message format for the first logged error
            first_call = mock_logger.error.call_args_list[0]
            args = first_call[0]
            assert "Background task raised an exception" in args[0]
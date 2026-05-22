"""Unit tests for the redesigned worker with AdmissionTracker.

Tests cover:
1. AdmissionTracker budget accounting (admit/release/progressive update)
2. Compound metric mode (max of gpu and decode fractions)
3. estimate_request_cost() for all 4 metric types
4. Reactive vs predictive strategy logic
5. Budget leak prevention via release
"""
import sys
import os
import asyncio
import unittest
from unittest.mock import MagicMock

# Mock external dependencies before importing worker
sys.modules['aio_pika'] = MagicMock()
sys.modules['aio_pika.abc'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['redis.asyncio'] = MagicMock()
sys.modules['httpx'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'worker'))
import worker
from worker import (
    AdmissionTracker, BottleneckMetric, estimate_request_cost,
    admit_static, admit_reactive, admit_predictive,
    CHARS_PER_TOKEN, KV_BYTES_PER_TOKEN,
)


def run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAdmissionTrackerDecode(unittest.TestCase):
    """Test AdmissionTracker with decode_throughput metric."""

    def setUp(self):
        self.tracker = AdmissionTracker(
            metric=BottleneckMetric.DECODE_THROUGHPUT,
            decode_capacity=100.0,  # 100 tokens capacity
            gpu_capacity=500.0,
            kv_capacity=1_000_000,
        )

    def test_initial_state(self):
        self.assertEqual(self.tracker.current_load(), 0.0)
        self.assertEqual(len(self.tracker.running), 0)

    def test_admit_increases_budget(self):
        run(self.tracker.admit("job1", {"decode": 50.0}, 50))
        self.assertAlmostEqual(self.tracker.current_load(), 0.5)
        self.assertEqual(len(self.tracker.running), 1)

    def test_admit_until_capacity(self):
        run(self.tracker.admit("job1", {"decode": 50.0}, 50))
        run(self.tracker.admit("job2", {"decode": 50.0}, 50))
        self.assertAlmostEqual(self.tracker.current_load(), 1.0)

    def test_release_frees_budget(self):
        run(self.tracker.admit("job1", {"decode": 50.0}, 50))
        run(self.tracker.release("job1"))
        self.assertAlmostEqual(self.tracker.current_load(), 0.0)
        self.assertEqual(len(self.tracker.running), 0)

    def test_release_unknown_job_is_safe(self):
        # Should not crash or corrupt state
        run(self.tracker.release("nonexistent"))
        self.assertAlmostEqual(self.tracker.current_load(), 0.0)

    def test_projected_load(self):
        run(self.tracker.admit("job1", {"decode": 50.0}, 50))
        # Current = 50%, projected with 30 more = 80%
        self.assertAlmostEqual(self.tracker.projected_load({"decode": 30.0}), 0.8)

    def test_update_progress_decreases_decode(self):
        run(self.tracker.admit("job1", {"decode": 100.0}, 100))
        self.assertAlmostEqual(self.tracker.current_load(), 1.0)
        # Generate 50 tokens — should free 50 from decode budget
        for i in range(1, 51):
            run(self.tracker.update_progress("job1", i))
        self.assertAlmostEqual(self.tracker.decode_used, 50.0)
        self.assertAlmostEqual(self.tracker.current_load(), 0.5)

    def test_budget_never_goes_negative(self):
        run(self.tracker.admit("job1", {"decode": 10.0}, 10))
        # Generate more tokens than estimated
        for i in range(1, 20):
            run(self.tracker.update_progress("job1", i))
        self.assertGreaterEqual(self.tracker.decode_used, 0.0)


class TestAdmissionTrackerGPU(unittest.TestCase):
    """Test AdmissionTracker with gpu_compute metric."""

    def setUp(self):
        self.tracker = AdmissionTracker(
            metric=BottleneckMetric.GPU_COMPUTE,
            decode_capacity=100.0,
            gpu_capacity=500.0,  # 500 tokens prefill capacity
            kv_capacity=1_000_000,
        )

    def test_gpu_load(self):
        run(self.tracker.admit("job1", {"gpu": 250.0}, 100))
        self.assertAlmostEqual(self.tracker.current_load(), 0.5)

    def test_gpu_released_after_first_token(self):
        """GPU budget should be released after prefill (first token generated)."""
        run(self.tracker.admit("job1", {"gpu": 250.0}, 100))
        self.assertAlmostEqual(self.tracker.gpu_used, 250.0)
        # First token = prefill done
        run(self.tracker.update_progress("job1", 1))
        self.assertAlmostEqual(self.tracker.gpu_used, 0.0)
        self.assertAlmostEqual(self.tracker.current_load(), 0.0)

    def test_gpu_not_released_before_first_token(self):
        """GPU budget stays until first token is generated."""
        run(self.tracker.admit("job1", {"gpu": 250.0}, 100))
        # No tokens generated yet — GPU budget should stay
        self.assertAlmostEqual(self.tracker.gpu_used, 250.0)


class TestAdmissionTrackerKVCache(unittest.TestCase):
    """Test AdmissionTracker with kv_cache metric."""

    def setUp(self):
        self.tracker = AdmissionTracker(
            metric=BottleneckMetric.KV_CACHE,
            decode_capacity=100.0,
            gpu_capacity=500.0,
            kv_capacity=1_000_000,  # 1MB KV budget
        )

    def test_kv_load(self):
        run(self.tracker.admit("job1", {"kv": 500_000}, 100))
        self.assertAlmostEqual(self.tracker.current_load(), 0.5)

    def test_kv_not_released_during_generation(self):
        """KV cache stays allocated until release — no progressive free."""
        run(self.tracker.admit("job1", {"kv": 500_000}, 100))
        for i in range(1, 50):
            run(self.tracker.update_progress("job1", i))
        # KV should still be fully allocated
        self.assertAlmostEqual(self.tracker.kv_used, 500_000)
        self.assertAlmostEqual(self.tracker.current_load(), 0.5)


class TestAdmissionTrackerCompound(unittest.TestCase):
    """Test compound metric — max(gpu_fraction, decode_fraction)."""

    def setUp(self):
        self.tracker = AdmissionTracker(
            metric=BottleneckMetric.COMPOUND,
            decode_capacity=100.0,
            gpu_capacity=100.0,
            kv_capacity=1_000_000,
        )

    def test_compound_takes_max(self):
        """Compound load = max of the two fractions."""
        run(self.tracker.admit("job1", {"gpu": 30.0, "decode": 70.0}, 70))
        # GPU = 30%, decode = 70% → compound = 70%
        self.assertAlmostEqual(self.tracker.current_load(), 0.7)

    def test_compound_gpu_dominant(self):
        run(self.tracker.admit("job1", {"gpu": 90.0, "decode": 20.0}, 20))
        # GPU = 90%, decode = 20% → compound = 90%
        self.assertAlmostEqual(self.tracker.current_load(), 0.9)

    def test_compound_projected(self):
        run(self.tracker.admit("job1", {"gpu": 30.0, "decode": 30.0}, 30))
        # Current: gpu=30%, decode=30% → 30%
        # Add: gpu=50, decode=10 → projected: gpu=80%, decode=40% → 80%
        projected = self.tracker.projected_load({"gpu": 50.0, "decode": 10.0})
        self.assertAlmostEqual(projected, 0.8)

    def test_compound_gpu_released_after_prefill(self):
        """In compound mode, GPU frees after first token but decode stays."""
        run(self.tracker.admit("job1", {"gpu": 50.0, "decode": 50.0}, 50))
        self.assertAlmostEqual(self.tracker.current_load(), 0.5)
        # First token — GPU released
        run(self.tracker.update_progress("job1", 1))
        # GPU = 0%, decode = 49/100 = 49%
        self.assertAlmostEqual(self.tracker._gpu_fraction(), 0.0)
        self.assertAlmostEqual(self.tracker._decode_fraction(), 0.49)


class TestEstimateRequestCost(unittest.TestCase):
    """Test cost estimation for different metric types."""

    def test_decode_throughput(self):
        costs = estimate_request_cost("Hello world", 100, BottleneckMetric.DECODE_THROUGHPUT)
        self.assertEqual(costs, {"decode": 100.0})

    def test_gpu_compute(self):
        prompt = "x" * 400  # 400 chars = 100 tokens at 4 chars/token
        costs = estimate_request_cost(prompt, 200, BottleneckMetric.GPU_COMPUTE)
        self.assertAlmostEqual(costs["gpu"], 100.0)

    def test_kv_cache(self):
        prompt = "x" * 40  # 40 chars = 10 tokens
        costs = estimate_request_cost(prompt, 90, BottleneckMetric.KV_CACHE)
        # Total tokens = 10 + 90 = 100, KV cost = 100 * KV_BYTES_PER_TOKEN
        expected = 100 * KV_BYTES_PER_TOKEN
        self.assertAlmostEqual(costs["kv"], expected)

    def test_compound(self):
        prompt = "x" * 400  # 100 tokens
        costs = estimate_request_cost(prompt, 200, BottleneckMetric.COMPOUND)
        self.assertAlmostEqual(costs["gpu"], 100.0)
        self.assertEqual(costs["decode"], 200.0)


class TestStrategyLogic(unittest.TestCase):
    """Test reactive vs predictive admission decisions."""

    def setUp(self):
        self.tracker = AdmissionTracker(
            metric=BottleneckMetric.DECODE_THROUGHPUT,
            decode_capacity=100.0,
            gpu_capacity=500.0,
            kv_capacity=1_000_000,
        )
        # Pre-load tracker to 70% capacity
        run(self.tracker.admit("existing", {"decode": 70.0}, 70))

    def test_static_always_admits(self):
        admitted, reason = run(admit_static(self.tracker, "Hello", 200))
        self.assertTrue(admitted)
        self.assertEqual(reason, "")

    def test_reactive_admits_below_threshold(self):
        """At 70% load with 80% threshold, reactive admits."""
        worker.ADMISSION_THRESHOLD = 0.80
        admitted, _ = run(admit_reactive(self.tracker, "Hello", 200))
        self.assertTrue(admitted)

    def test_reactive_rejects_above_threshold(self):
        """At 70% load with 60% threshold, reactive rejects."""
        worker.ADMISSION_THRESHOLD = 0.60
        admitted, reason = run(admit_reactive(self.tracker, "Hello", 200))
        self.assertFalse(admitted)
        self.assertIn("70.0%", reason)

    def test_reactive_ignores_request_size(self):
        """Reactive doesn't care if request is tiny or huge — same decision."""
        worker.ADMISSION_THRESHOLD = 0.80
        small_admitted, _ = run(admit_reactive(self.tracker, "Hi", 5))
        large_admitted, _ = run(admit_reactive(self.tracker, "x" * 4000, 2000))
        self.assertEqual(small_admitted, large_admitted)

    def test_predictive_admits_small_request(self):
        """At 70% load, a 5-token request → projected 75% < 80% threshold."""
        worker.ADMISSION_THRESHOLD = 0.80
        worker.BOTTLENECK_METRIC = "decode_throughput"
        admitted, _ = run(admit_predictive(self.tracker, "Hello", 5))
        self.assertTrue(admitted)

    def test_predictive_rejects_large_request(self):
        """At 70% load, a 200-token request → projected 270% > 80% threshold."""
        worker.ADMISSION_THRESHOLD = 0.80
        worker.BOTTLENECK_METRIC = "decode_throughput"
        admitted, reason = run(admit_predictive(self.tracker, "Hello", 200))
        self.assertFalse(admitted)

    def test_predictive_differentiates_by_size(self):
        """Key test: same load, but predictive treats small and large differently."""
        worker.ADMISSION_THRESHOLD = 0.80
        worker.BOTTLENECK_METRIC = "decode_throughput"
        small_admitted, _ = run(admit_predictive(self.tracker, "Hello", 5))
        large_admitted, _ = run(admit_predictive(self.tracker, "Hello", 200))
        self.assertTrue(small_admitted, "Small request should be admitted at 70% load")
        self.assertFalse(large_admitted, "Large request should be rejected at 70% load")

    def tearDown(self):
        # Reset globals
        worker.ADMISSION_THRESHOLD = 0.80
        worker.BOTTLENECK_METRIC = "decode_throughput"


class TestBudgetLeakPrevention(unittest.TestCase):
    """Verify that tracker.release() prevents budget leaks."""

    def test_release_after_admit_restores_budget(self):
        tracker = AdmissionTracker(
            metric=BottleneckMetric.DECODE_THROUGHPUT,
            decode_capacity=100.0,
            gpu_capacity=500.0,
            kv_capacity=1_000_000,
        )
        run(tracker.admit("job1", {"decode": 50.0}, 50))
        self.assertAlmostEqual(tracker.current_load(), 0.5)

        # Simulate error — release without completion
        run(tracker.release("job1"))
        self.assertAlmostEqual(tracker.current_load(), 0.0)
        self.assertEqual(len(tracker.running), 0)

    def test_double_release_is_safe(self):
        """Releasing the same job twice should not corrupt state."""
        tracker = AdmissionTracker(
            metric=BottleneckMetric.DECODE_THROUGHPUT,
            decode_capacity=100.0,
            gpu_capacity=500.0,
            kv_capacity=1_000_000,
        )
        run(tracker.admit("job1", {"decode": 50.0}, 50))
        run(tracker.release("job1"))
        run(tracker.release("job1"))  # Second release
        self.assertAlmostEqual(tracker.current_load(), 0.0)

    def test_partial_progress_then_release(self):
        """After partial token generation + release, budget fully freed."""
        tracker = AdmissionTracker(
            metric=BottleneckMetric.DECODE_THROUGHPUT,
            decode_capacity=100.0,
            gpu_capacity=500.0,
            kv_capacity=1_000_000,
        )
        run(tracker.admit("job1", {"decode": 50.0}, 50))
        # Generate 20 tokens (decode_used drops from 50 to 30)
        for i in range(1, 21):
            run(tracker.update_progress("job1", i))
        self.assertAlmostEqual(tracker.decode_used, 30.0)

        # Release should free the remaining 30
        run(tracker.release("job1"))
        self.assertAlmostEqual(tracker.decode_used, 0.0)


class TestStats(unittest.TestCase):
    """Test the stats() method for logging."""

    def test_stats_structure(self):
        tracker = AdmissionTracker(
            metric=BottleneckMetric.COMPOUND,
            decode_capacity=100.0,
            gpu_capacity=500.0,
            kv_capacity=1_000_000,
        )
        run(tracker.admit("job1", {"gpu": 100.0, "decode": 30.0}, 30))
        stats = tracker.stats()
        self.assertEqual(stats["running_count"], 1)
        self.assertEqual(stats["decode_used"], 30.0)
        self.assertEqual(stats["gpu_used"], 100.0)
        self.assertIn("utilization_pct", stats)


if __name__ == "__main__":
    unittest.main()

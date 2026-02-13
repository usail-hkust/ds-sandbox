"""
Test streaming callbacks for ds-sandbox.
"""

import asyncio
from ds_sandbox import Sandbox
from ds_sandbox.config import SandboxConfig


async def test_streaming_callbacks():
    """Test streaming callbacks with Jupyter execution."""
    # Enable Jupyter for chart support
    config = SandboxConfig(
        use_jupyter=True,
        default_timeout_sec=60,
        workspace_base_dir="/tmp/ds_sandbox_workspaces",
    )

    sandbox = await Sandbox.create_async(config=config)

    print("=== Testing Streaming Callbacks ===\n")

    # Test stdout callback
    print("1. Testing on_stdout callback:")
    stdout_data = []

    def on_stdout(data):
        stdout_data.append(data)
        print(f"  [stdout] {data}", end="")

    code = """
print("Hello")
print("World")
import pandas as pd
print("pandas loaded")
"""

    result = await sandbox.run_code_async(
        code,
        on_stdout=on_stdout,
    )

    print(f"\n  Collected {len(stdout_data)} stdout chunks")
    print(f"  Final stdout: {result.stdout}")
    print(f"  Success: {result.success}\n")

    # Test stderr callback
    print("2. Testing on_stderr callback:")
    stderr_data = []

    def on_stderr(data):
        stderr_data.append(data)
        print(f"  [stderr] {data}", end="")

    code_with_error = """
import sys
print("Before error", file=sys.stderr)
print("After error")
"""

    result2 = await sandbox.run_code_async(
        code_with_error,
        on_stderr=on_stderr,
    )

    print(f"\n  Collected {len(stderr_data)} stderr chunks")
    print(f"  Final stderr: {result2.stderr}")
    print(f"  Success: {result2.success}\n")

    # Test on_result callback (charts)
    print("3. Testing on_result callback (charts):")
    result_data = []

    def on_result(data):
        result_data.append(data)
        if 'text' in data:
            print(f"  [result] text: {data['text'][:50]}...")
        elif 'png' in data:
            print(f"  [result] png: {len(data['png'])} bytes")

    chart_code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Chart")
"""

    result3 = await sandbox.run_code_async(
        chart_code,
        on_result=on_result,
    )

    print(f"\n  Collected {len(result_data)} results")
    print(f"  Results in CodeResult: {result3.results}")
    print(f"  Success: {result3.success}\n")

    # Test all callbacks together
    print("4. Testing all callbacks together:")
    all_stdout = []
    all_stderr = []
    all_results = []

    mixed_code = """
print("Starting...")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1,2,3], [1,4,9])
print("Chart created")
"""

    result4 = await sandbox.run_code_async(
        mixed_code,
        on_stdout=lambda d: all_stdout.append(d),
        on_stderr=lambda d: all_stderr.append(d),
        on_result=lambda d: all_results.append(d),
    )

    print(f"  stdout chunks: {len(all_stdout)}")
    print(f"  stderr chunks: {len(all_stderr)}")
    print(f"  result chunks: {len(all_results)}")
    print(f"  Success: {result4.success}")

    await sandbox.kill()
    print("\n=== Test Complete ===")


async def test_sync_interface():
    """Test synchronous interface with callbacks."""
    print("\n=== Testing Sync Interface ===\n")

    config = SandboxConfig(
        use_jupyter=True,
        workspace_base_dir="/tmp/ds_sandbox_workspaces",
    )
    sandbox = await Sandbox.create_async(config=config)
    stdout_data = []

    result = await sandbox.run_code_async(
        "print('sync test')",
        on_stdout=lambda d: stdout_data.append(d),
    )

    print(f"  Sync stdout: {result.stdout}")
    print(f"  Callback received: {len(stdout_data) > 0}")

    await sandbox.kill()
    print("=== Sync Test Complete ===")


if __name__ == "__main__":
    # Run async test
    asyncio.run(test_streaming_callbacks())

    # Run sync test
    asyncio.run(test_sync_interface())

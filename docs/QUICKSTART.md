# ds-sandbox Quick Start

ds-sandbox is a local sandbox execution environment with code execution and file operations, compatible with E2B API.

## Features

- **Two Execution Modes**: local (local process) and docker (container isolation)
- **E2B Compatible API**: Interface design aligned with E2B for easy migration
- **Pre-installed Data Science Libraries**: pandas, matplotlib, scikit-learn, etc.
- **File Operations**: Support for reading and writing files
- **Code Execution**: Support for Python code execution

## Installation

```bash
pip install ds-sandbox
```

## Quick Start

### 1. Basic Usage (Auto-create workspace)

```python
from ds_sandbox import Sandbox

# Create sandbox (synchronous)
sandbox = Sandbox.create()

# Execute code
result = sandbox.run_code("print('hello world')")
print(result.stdout)  # hello world

# Kill sandbox
sandbox.kill()
```

### 2. Async Usage

```python
import asyncio
from ds_sandbox import Sandbox

async def main():
    sandbox = await Sandbox.create_async()

    result = await sandbox.run_code_async("print('hello')")

    await sandbox.kill()

asyncio.run(main())
```

### 3. File Operations

```python
sandbox = Sandbox.create()

# Write file
sandbox.files.write("data.csv", "col1,col2\n1,2")

# Read file
content = sandbox.files.read("data.csv")
print(content)

sandbox.kill()
```

## Two Modes Detailed

### Local Mode

Local mode executes code in the local process, directly using the host filesystem.

```python
from ds_sandbox import Sandbox, SandboxConfig

config = SandboxConfig(default_backend="local")
sandbox = Sandbox.create(config=config)

result = sandbox.run_code("print('local mode')")
sandbox.kill()
```

**Characteristics**:
- Fast execution (no container overhead)
- Shared filesystem with host
- Suitable for local development and testing

### Docker Mode

Docker mode executes code in containers, providing stronger isolation.

```python
from ds_sandbox import Sandbox, SandboxConfig

config = SandboxConfig(default_backend="docker")
sandbox = Sandbox.create(config=config)

result = sandbox.run_code("print('docker mode')")
sandbox.kill()
```

**Characteristics**:
- Container isolation, more secure
- Independent filesystem
- Suitable for production environments

## DSLighting Integration

ds-sandbox can be used as a sandbox backend for DSLighting.

### Local Mode (Soft Links)

When DSLighting uses local mode, it shares data directories via soft links without copying files:

```python
# DSLighting calls automatically use soft links
# DSLighting workspace: /path/to/workspace/sandbox/
# ds-sandbox workspace: /tmp/ds_sandbox_workspaces/ws-xxx/ -> soft link to DSLighting directory
```

Configure environment variables:
```bash
export DSLIGHTING_SANDBOX_BACKEND=ds_sandbox
export DSLIGHTING_SANDBOX_BACKEND_TYPE=local
```

### Docker Mode (Upload Files)

In Docker mode, DSLighting automatically uploads files to the container:

```bash
export DSLIGHTING_SANDBOX_BACKEND=ds_sandbox
export DSLIGHTING_SANDBOX_BACKEND_TYPE=docker
```

## Configuration Options

### SandboxConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| default_backend | str | "auto" | Execution backend: local, docker, auto |
| workspace_base_dir | str | "/tmp/ds_sandbox_workspaces" | Workspace root directory |
| use_jupyter | bool | False | Use Jupyter execution (supports charts) |

### Execution Parameters

```python
result = sandbox.run_code(
    code="print('hello')",
    timeout=60,  # Timeout in seconds
    on_stdout=lambda x: print(x),  # stdout callback
    on_stderr=lambda x: print(x),  # stderr callback
    on_result=lambda x: print(x),  # Result callback (charts, etc.)
)
```

## Streaming Callbacks

Support for real-time execution output:

```python
sandbox = Sandbox.create(config=SandboxConfig(use_jupyter=True))

# Use callbacks to get real-time output
result = sandbox.run_code(
    """
import matplotlib.pyplot as plt
plt.plot([1,2,3], [1,4,9])
print("Chart created!")
""",
    on_stdout=lambda x: print(f"[stdout] {x}"),
    on_stderr=lambda x: print(f"[stderr] {x}"),
    on_result=lambda x: print(f"[result] {x}"),  # Chart data
)

sandbox.kill()
```

## Error Handling

```python
try:
    result = sandbox.run_code("import non_existent_module")
    if not result.success:
        print(f"Error: {result.error}")
except Exception as e:
    print(f"Exception: {e}")
```

## Complete Example

```python
from ds_sandbox import Sandbox, SandboxConfig

# Create configuration
config = SandboxConfig(
    default_backend="local",
    workspace_base_dir="/tmp/my_workspaces",
)

# Create sandbox
sandbox = Sandbox.create(config=config)

# Write data file
sandbox.files.write("train.csv", "id,value\n1,100\n2,200")

# Execute code
result = sandbox.run_code("""
import pandas as pd
df = pd.read_csv('train.csv')
print(df.sum())
""")

print(result.stdout)

# Cleanup
sandbox.kill()
```

## Notes

1. **Local Mode**: Code executes in the host process, pay attention to resource isolation
2. **Docker Mode**: Requires Docker runtime permissions
3. **File Paths**: Use relative paths when writing files, they will be created in the workspace directory
4. **Timeout**: Default timeout is 3600 seconds, adjust as needed

## Troubleshooting

### Permission Errors

```bash
# Ensure you have permission to create workspace directory
mkdir -p /tmp/ds_sandbox_workspaces
chmod 777 /tmp/ds_sandbox_workspaces
```

### Docker Mode Failure

```bash
# Check if Docker is running
docker ps

# Check ds-sandbox image
docker images | grep ds-sandbox
```

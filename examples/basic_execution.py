# Basic usage example

import asyncio
from ds_sandbox import SandboxSDK

async def main():
    # Initialize SDK (will use default config)
    sdk = SandboxSDK()

    try:
        # Create a workspace
        workspace = await sdk.create_workspace("demo-001")

        print(f"✓ Workspace created: {workspace.workspace_id}")
        print(f"  Host path: {workspace.host_path}")

        # Execute code (using relative paths)
        result = await sdk.execute(
            workspace_id="demo-001",
            code="""
import pandas as pd
import os

# Check current directory
print(f"Working directory: {os.getcwd()}")

# List files
print(f"Files: {sorted(os.listdir('.'))}")

# Read data from relative path
df = pd.read_csv('data/train.csv')

# Show first few rows
print(df.head())

print(f"Total rows: {len(df)}")
""",
            mode="safe",
            timeout_sec=300
        )

        if result.success:
            print("\n✅ Execution successful!")
            print(f"Output: {result.stdout}")
        else:
            print("\n❌ Execution failed!")
            print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"\n❗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

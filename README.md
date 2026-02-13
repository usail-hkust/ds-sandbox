<div align="center">

<img src="assets/ds-sandbox.png" alt="ds-sandbox Logo" width="180" style="border-radius: 15px;">

# ds-sandbox

> Workspace-first AI code execution sandbox

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/PyPI-ds--sandbox-blue?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/ds-sandbox/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-usail--hkust%2Fds--sandbox-blue?style=flat-square&logo=github)](https://github.com/usail-hkust/ds-sandbox)

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/🚀-Quick_Start-green?style=for-the-badge" alt="Quick Start"></a>
  &nbsp;&nbsp;
  <a href="#-features"><img src="https://img.shields.io/badge/⚡-Features-blue?style=for-the-badge" alt="Features"></a>
  &nbsp;&nbsp;
  <a href="#-documentation"><img src="https://img.shields.io/badge/📚-Docs-orange?style=for-the-badge" alt="Documentation"></a>
  &nbsp;&nbsp;
  <a href="#-quick-start"><img src="https://img.shields.io/badge/📖-User_Guide-purple?style=for-the-badge" alt="User Guide"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/usail-hkust/ds-sandbox/stargazers"><img src="https://img.shields.io/github/stars/usail-hkust/ds-sandbox?style=for-the-badge" alt="Stars"></a>
  &nbsp;&nbsp;
  <img src="https://komarev.com/ghpvc/?username=usail-hkust&repo=ds-sandbox&style=for-the-badge" alt="Profile views">
</p>

</div>

<div align="center">

🔧 **Pluggable Backends** &nbsp;•&nbsp; 🛡️ **Enterprise Security** &nbsp;•&nbsp; 🌐 **Multi-Language Support** (coming soon) &nbsp;•&nbsp; 🚀 **Production Ready**

[⭐ Star us](https://github.com/usail-hkust/ds-sandbox/stargazers) &nbsp;•&nbsp; [💬 Discussions](https://github.com/usail-hkust/ds-sandbox/discussions)

</div>

---

## ✨ Features

- 🔧 **Pluggable Backends** - Docker, Local Subprocess, Firecracker (coming soon), Kata Containers (coming soon)
- 📁 **Workspace-First** - Data in workspace relative paths, seamless data access
- 🛡️ **Enterprise Security** - Code scanning, resource limits, network policies, audit logging
- 🚀 **Production Ready** - Metrics, health checks, high availability
- 📋 **E2B-Compatible API** - Templates, user/workdir support for easy migration

## 🚀 Why ds-sandbox?

ds-sandbox is lightweight, local-first, and perfect for individual developers or small-scale deployments:

- 🖥️ **Local Deployment** - No cloud services or API keys required, runs entirely on your machine
- 🐳 **Simple Setup** - Single-machine or Docker deployment, no complex infrastructure
- 💰 **Free & Self-Hosted** - No cloud costs, full control over your data and execution environment
- 🔒 **Data Privacy** - Code and data never leave your infrastructure

**Note:** ds-sandbox only supports Python code execution. JavaScript/Node.js execution is not supported.

## 🎯 Use Cases

- **AI Agent Code Execution** - Safe execution of LLM-generated code
- **Data Science Workflows** - Model training, data analysis, visualization
- **Multi-tenant Environments** - Isolated workspaces for multiple users
- **Automated ML Pipelines** - Batch processing with workspace persistence

## 📦 Installation

### From PyPI (coming soon)

```bash
pip install ds-sandbox
```

### From Source

```bash
git clone https://github.com/usail-hkust/ds-sandbox.git
cd ds-sandbox
pip install -e .
```

## 🚀 Quick Start

Write code for starting Sandbox (local mode by default), executing Python inside it and listing all files inside the root directory.

> **Note**: Local mode uses the same Python environment as your local machine. For isolated container execution, use Docker mode.

### Python

```python
# main.py
from ds_sandbox import Sandbox

sbx = Sandbox.create()  # Create a new sandbox instance (local mode by default)
execution = sbx.run_code("print('hello world')")  # Execute Python inside the sandbox
print(execution.logs)

files = sbx.files.list("/")
print(files)

sbx.kill()
```

### Connect LLMs

ds-sandbox can work with any LLM and AI framework. The easiest way to connect an LLM to ds-sandbox is to use the tool use capabilities of the LLM (sometimes known as function calling).

#### OpenAI

```python
# pip install openai ds-sandbox
from openai import OpenAI
from ds_sandbox import Sandbox

# Create OpenAI client
client = OpenAI()
system = "You are a helpful assistant that can execute python code in a Jupyter notebook. Only respond with the code to be executed and nothing else. Strip backticks in code blocks."
prompt = "Calculate how many r's are in the word 'strawberry'"

# Send messages to OpenAI API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
)

# Extract the code from the response
code = response.choices[0].message.content

# Execute code in ds-sandbox
if code:
    with Sandbox.create() as sandbox:
        execution = sandbox.run_code(code)
        result = execution.text

    print(result)
```

### Upload & Download Files

ds-sandbox allows you to upload and download files to and from the Sandbox.

**Note:** The default working directory is `/home/user`. Files are stored relative to the workspace root.

#### Upload File

```python
from ds_sandbox import Sandbox

sbx = Sandbox.create()

# Read local file relative to the current working directory
with open("local/file", "rb") as file:
    # Upload file to the sandbox (default workdir is /home/user)
    sbx.files.write("my-file", file)

sbx.kill()
```

#### Upload Multiple Files

```python
from ds_sandbox import Sandbox

sbx = Sandbox.create()

# Read local file relative to the current working directory
with open("local/file/a", "rb") as file:
    # Upload file to the sandbox
    sbx.files.write("my-file-a", file)

with open("local/file/b", "rb") as file:
    # Upload file to the sandbox
    sbx.files.write("my-file-b", file)

sbx.kill()
```

#### Download File

```python
from ds_sandbox import Sandbox

sbx = Sandbox.create()

# Download file from the sandbox
content = sbx.files.read('my-file')
# Write file to local path relative to the current working directory
with open('local/file', 'w') as file:
    file.write(content)
```

#### Download Multiple Files

```python
from ds_sandbox import Sandbox

sbx = Sandbox.create()

# Download file A from the sandbox
contentA = sbx.files.read('my-file-a')
# Write file A to local path relative to the current working directory
with open('local/file/a', 'w') as file:
    file.write(contentA)

# Download file B from the sandbox
contentB = sbx.files.read('my-file-b')
# Write file B to local path relative to the current working directory
with open('local/file/b', 'w') as file:
    file.write(contentB)

sbx.kill()
```

### Templates

ds-sandbox supports E2B-compatible templates for customizing sandbox environments.

```python
from ds_sandbox import Sandbox, Template

# Create a custom template
template = Template(
    id="my-custom-template",
    name="My Custom Template",
    env={"CUSTOM_VAR": "value", "PYTHONPATH": "/home/user/lib"},
    files={
        "setup.sh": "#!/bin/bash\necho 'Setting up...'\n",
        "config.json": '{"key": "value"}',
    },
    user="developer",
    workdir="/home/developer",
)

# Use the template when creating a sandbox
sandbox = Sandbox.create(template=template)

# Access user and workdir
print(sandbox.user)    # "developer"
print(sandbox.workdir) # "/home/developer"

sandbox.kill()
```

Templates can also be stored as YAML files in the templates directory (default: `/etc/ds-sandbox/templates`):

```yaml
# /etc/ds-sandbox/templates/my-template.yaml
id: "my-template"
name: "My Custom Template"
env:
  CUSTOM_VAR: "value"
files:
  setup.sh: "#!/bin/bash\necho 'Hello'\n"
user: "developer"
workdir: "/home/developer"
```

Load templates from files:

```python
from ds_sandbox import Sandbox, SandboxConfig

# Load template from file
template = SandboxConfig.load_template("my-template")
if template:
    sandbox = Sandbox.create(template=template)
```

### User and Working Directory

ds-sandbox provides default user "user" and working directory "/home/user":

```python
from ds_sandbox import Sandbox

sandbox = Sandbox.create()

# Default user and workdir
print(sandbox.user)    # "user"
print(sandbox.workdir) # "/home/user"

sandbox.kill()
```

You can customize user and workdir via Template or SandboxConfig:

```python
from ds_sandbox import Sandbox, Template, SandboxConfig

# Via template
template = Template(user="customuser", workdir="/workspace")
sandbox = Sandbox.create(template=template)

# Via config (sets defaults for all sandboxes)
config = SandboxConfig(default_user="admin", default_workdir="/home/admin")
sandbox = Sandbox.create(config=config)
```

### Internet Access Configuration

ds-sandbox allows you to configure internet access for sandbox execution. You can control network access at both the request level and the configuration level.

#### Request-Level Configuration

```python
from ds_sandbox import Sandbox

sandbox = Sandbox.create()

# Allow internet access (default)
execution = sandbox.run_code(
    "import requests; print(requests.get('https://api.github.com').status_code)",
    allow_internet=True,
    network_policy="allow"
)

# Deny all internet access
execution = sandbox.run_code(
    "import requests; print(requests.get('https://api.github.com').status_code)",
    allow_internet=False,
    network_policy="deny"
)

# Whitelist specific domains
execution = sandbox.run_code(
    "import requests; print(requests.get('https://api.github.com').status_code)",
    allow_internet=True,
    network_policy="whitelist",
    network_whitelist=["api.github.com", "pypi.org"]
)

sandbox.kill()
```

#### Configuration-Level Defaults

You can set default network configuration via SandboxConfig:

```python
from ds_sandbox import Sandbox, SandboxConfig

# Set default network policy
config = SandboxConfig(
    allow_internet=True,
    default_network_policy="allow",  # "allow", "deny", "whitelist"
    network_whitelist=["api.github.com"]  # Only allow these domains
)
sandbox = Sandbox.create(config=config)
```

#### Environment Variables

You can also configure defaults via environment variables:

```bash
export SANDBOX_ALLOW_INTERNET=true
export SANDBOX_NETWORK_POLICY=allow
export SANDBOX_NETWORK_WHITELIST=api.github.com,pypi.org
```

**Network Policy Options:**
- `allow` - Allow all internet access (default)
- `deny` - Block all internet access
- `whitelist` - Only allow access to specified domains/IPs

### Python Only

ds-sandbox only supports Python code execution. JavaScript/Node.js execution is not supported.

```python
# This works
result = sandbox.run_code("print('Hello, World!')")

# JavaScript/Node.js is NOT supported
# result = sandbox.run_js("console.log('Hello')")  # Won't work!
```

## 📚 Documentation

- [Quick Start](docs/QUICKSTART.md)

## 🏗️ Project Structure

```
ds-sandbox/
├── src/ds_sandbox/          # Source code
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                 # Example code
└── deployment/               # Deployment configs
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

Apache-2.0

## 🙏 Acknowledgments

Built upon research from existing sandbox solutions including:
- [e2b.dev](https://e2b.dev) - Inspiration for API design
- [Firecracker](https://github.com/firecracker-microvm/firecracker) - MicroVM technology (coming soon)
- [gVisor](https://gvisor.dev) - User-space kernel (coming soon)
- [Kata Containers](https://katacontainers.io) - K8s sandbox (coming soon)
- [nbclient](https://github.com/jupyter/nbclient) - Notebook execution

---

## ⭐ Star History

<div align="center">

[![Stargazers repo roster for @usail-hkust/ds-sandbox](https://reporoster.com/stars/usail-hkust/ds-sandbox)](https://github.com/usail-hkust/ds-sandbox/stargazers)

[![Forkers repo roster for @usail-hkust/ds-sandbox](https://reporoster.com/forks/usail-hkust/ds-sandbox)](https://github.com/usail-hkust/ds-sandbox/network/members)

[![Star History Chart](https://api.star-history.com/svg?repos=usail-hkust/ds-sandbox&type=Date)](https://star-history.com/#usail-hkust/ds-sandbox&Date)

</div>

---

## 💬 WeChat Community

Join our WeChat group to connect with other users and developers!

<div align="center">

<img src="assets/wechat_group.jpg" alt="WeChat Group" width="300" style="border-radius: 10px; border: 2px solid #e0e0e0;">

**Scan the QR code above to join the ds-sandbox user community**

</div>

In the group, you can:
- 🤝 Connect with other users and share experiences
- 💡 Suggest features and provide feedback
- 🐛 Report bugs and get help
- 📢 Stay updated with the latest development news

---

## 🔗 Community

<div align="center">

**[ds-sandbox Community](https://github.com/usail-hkust/ds-sandbox)**

[💬 WeChat Group](#-wechat-community) · [⭐ Star us](https://github.com/usail-hkust/ds-sandbox/stargazers) · [🐛 Report a bug](https://github.com/usail-hkust/ds-sandbox/issues) · [💬 Discussions](https://github.com/usail-hkust/ds-sandbox/discussions)

</div>

---

## 📊 Project Statistics

![](https://komarev.com/ghpvc/?username=usail-hkust&repo=ds-sandbox&style=for-the-badge)
![](https://img.shields.io/github/issues/usail-hkust/ds-sandbox?style=for-the-badge)
![](https://img.shields.io/github/forks/usail-hkust/ds-sandbox?style=for-the-badge)
![](https://img.shields.io/github/stars/usail-hkust/ds-sandbox?style=for-the-badge)

---

## 📚 Citation

If you use ds-sandbox in your research, please cite:

```bibtex
@software{ds-sandbox2025,
  title = {ds-sandbox: Workspace-first AI Code Execution Sandbox},
  author = {Liu, F. and others},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/usail-hkust/ds-sandbox},
  version = {1.0.0}
}
```

Or in plain text:

```
Liu, F., et al. (2025). ds-sandbox: Workspace-first AI Code Execution Sandbox.
GitHub repository. https://github.com/usail-hkust/ds-sandbox
```

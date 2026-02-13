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
- 🌐 **Multi-Language Support** (coming soon) - REST API, Python SDK, MCP server
- 🚀 **Production Ready** - Metrics, health checks, high availability

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

Write code for starting Sandbox, executing Python inside it and listing all files inside the root directory.

### Python

```python
# main.py
from ds_sandbox import Sandbox

sbx = Sandbox()  # Create a new sandbox instance
execution = sbx.run_code("print('hello world')")  # Execute Python inside the sandbox
print(execution.logs)

files = sbx.files.list("/")
print(files)
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
    with Sandbox() as sandbox:
        execution = sandbox.run_code(code)
        result = execution.text

    print(result)
```

### Upload & Download Files

ds-sandbox allows you to upload and download files to and from the Sandbox.

#### Upload File

```python
from ds_sandbox import Sandbox

sbx = Sandbox()

# Read local file relative to the current working directory
with open("local/file", "rb") as file:
    # Upload file to the sandbox to absolute path '/home/user/my-file'
    sbx.files.write("/home/user/my-file", file)
```

#### Upload Multiple Files

```python
from ds_sandbox import Sandbox

sbx = Sandbox()

# Read local file relative to the current working directory
with open("local/file/a", "rb") as file:
    # Upload file to the sandbox to absolute path '/home/user/my-file-a'
    sbx.files.write("/home/user/my-file-a", file)

with open("local/file/b", "rb") as file:
    # Upload file to the sandbox to absolute path '/home/user/my-file-b'
    sbx.files.write("/home/user/my-file-b", file)
```

#### Download File

```python
from ds_sandbox import Sandbox

sbx = Sandbox()

# Download file from the sandbox to absolute path '/home/user/my-file'
content = sbx.files.read('/home/user/my-file')
# Write file to local path relative to the current working directory
with open('local/file', 'w') as file:
    file.write(content)
```

#### Download Multiple Files

```python
from ds_sandbox import Sandbox

sbx = Sandbox()

# Download file A from the sandbox by absolute path '/home/user/my-file-a'
contentA = sbx.files.read('/home/user/my-file-a')
# Write file A to local path relative to the current working directory
with open('local/file/a', 'w') as file:
    file.write(contentA)

# Download file B from the sandbox by absolute path '/home/user/my-file-b'
contentB = sbx.files.read('/home/user/my-file-b')
# Write file B to local path relative to the current working directory
with open('local/file/b', 'w') as file:
    file.write(contentB)
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

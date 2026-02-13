# ds-sandbox 完整技术方案

> **版本**: v1.0
> **日期**: 2026-02-12
> **状态**: 设计方案（最终版）

---

## 📋 目录

- [一、方案摘要](#一方案摘要)
- [二、核心原则](#二核心原则)
- [三、项目结构](#三项目结构)
- [四、核心模块](#四核心模块)
- [五、公共接口](#五公共接口)
- [六、安全设计](#六安全设计)
- [七、API设计](#七api设计)
- [八、SDK设计](#八sdk设计)
- [九、数据管理](#九数据管理)
- [十、测试策略](#十测试策略)
- [十一、里程碑](#十一里程碑)
- [十二、技术选型](#十二技术选型)

---

## 一、方案摘要

### 1.1 背景与问题

**当前现状**：
- ❌ 缺乏开源的生产级数据科学沙箱
- ❌ 现有方案要么太专用（e2b.dev），要么太通用（无数据管理）
- ❌ AI agent代码执行缺乏统一的workspace-first数据访问方案

**核心需求**：
```
输入：code + workspace + datasets
输出：ExecutionResult + artifacts
隔离：可配置（Docker/Firecracker/Kata）
接口：REST / Python SDK / MCP
```

### 1.2 目标定位

**项目定位**：
- 📦 **完全独立**的Python包（命名 `ds-sandbox`）
- 🎯 **通用目的** - 任何AI/ML项目可使用
- 🔧 **可插拔架构** - 支持多种隔离后端
- 📁 **Workspace优先** - 数据在工作目录中，相对路径访问
- 🌐 **多接口支持** - REST、Python SDK、MCP服务器

**不做什么**（明确边界）：
- ❌ **不实现** AutoML训练编排
- ❌ **不实现** 特定Agent工作流（AIDE、AutoMind等）
- ❌ **不绑定** 任何上层业务框架
- ✅ **只提供** 底层代码执行能力

### 1.3 核心价值

1. **统一抽象层** - 不同后端用统一接口
2. **Workspace原生** - 数据在workspace相对路径访问，符合DS习惯
3. **策略驱动** - 根据安全策略自动选择隔离级别
4. **生产就绪** - 审计、监控、限流、故障注入
5. **易于集成** - 三种接口适配不同使用场景

---

## 二、核心原则

### 2.1 Workspace-First原则

**定义**：
```python
# 用户代码视角
import pandas as pd
df = pd.read_csv('data/train.csv')      # ✅ 相对路径，直观
model.save('models/rf.pkl')              # ✅ 相对路径，可预测

# ❌ 不推荐
df = pd.read_csv('/data/bike/train.csv')  # 硬编码，难维护
df = pd.read_csv('/mnt/datasets/bike/train.csv')  # 绝对路径，不通用
```

**架构保证**：
```
Workspace结构：
/opt/workspaces/{workspace_id}/
  ├── data/          # 数据集（只读或读写）
  ├── models/        # 模型持久化
  ├── outputs/      # 输出文件
  └── .workspace/   # 元数据

Sandbox挂载：
host: /opt/workspaces/user-123  →  guest: /workspace

工作目录固定：/workspace
```

### 2.2 策略驱动安全

**自动隔离级别选择**：
```python
class SecurityContext(BaseModel):
    network_disabled: bool = True
    enable_gpu: bool = False

    @computed_field
    def recommended_isolation(self) -> str:
        if self.enable_gpu:
            return "secure"      # GPU需要VM隔离
        elif not self.network_disabled:
            return "secure"      # 网络访问需要VM
        return "fast"           # 默认用Docker
```

### 2.3 可插拔后端原则

**后端契约**：
```python
class SandboxBackend(ABC):
    @abstractmethod
    async def execute(
        self,
        request: ExecutionRequest,
        workspace: Workspace
    ) -> ExecutionResult:
        """执行代码并返回结果"""

    @abstractmethod
    async def health_check(self) -> BackendStatus:
        """后端健康检查"""
```

**支持的后端**：
1. **Docker** (~100ms启动) - 默认，快速迭代
2. **Firecracker** (~200ms启动) - 生产环境，强隔离
3. **Kata Containers** (~1s启动) - K8s原生，可选

---

## 三、项目结构

```
ds-sandbox/                                      # 项目根目录
├── pyproject.toml                               # 打包配置
├── README.md                                    # 项目说明
├── LICENSE                                      # Apache-2.0
├── PROPOSAL.md                                  # 本文档
│
├── src/ds_sandbox/                            # 源代码包
│   ├── __init__.py
│   │                                        # 公开API: SandboxManager, SandboxSDK
│   │
│   ├── config.py                              # 配置模型
│   ├── types.py                                # 公共类型定义
│   ├── errors.py                               # 异常定义
│   │
│   ├── manager.py                              # 🔑 核心管理器
│   │   # - 后端路由
│   │   # - 策略决策
│   │   # - 执行编排
│   │
│   ├── backends/                               # 隔离后端
│   │   ├── __init__.py
│   │   ├── base.py                              # 抽象接口
│   │   ├── docker.py                              # Docker实现
│   │   ├── firecracker.py                         # Firecracker实现（Phase 2）
│   │   ├── kata.py                               # Kata实现（Phase 2）
│   │   └── router.py                           # 后端路由器
│   │
│   ├── workspace/                              # Workspace管理
│   │   ├── __init__.py
│   │   ├── manager.py                           # Workspace生命周期
│   │   └── service.py                           # Workspace服务
│   │
│   ├── data/                                   # 数据管理
│   │   ├── __init__.py
│   │   ├── registry.py                           # 数据集注册表
│   │   ├── mounter.py                            # 挂载管理
│   │   └── catalog.py                            # 数据集目录
│   │
│   ├── storage/                                # 存储抽象
│   │   ├── __init__.py
│   │   ├── volumes.py                            # 卷管理
│   │   └── snapshots.py                          # 快照功能（Phase 4）
│   │
│   ├── security/                               # 安全层
│   │   ├── __init__.py
│   │   ├── policies.py                            # 网络策略
│   │   ├── scanner.py                            # 代码扫描
│   │   └── audit.py                             # 审计日志
│   │
│   ├── monitoring/                              # 监控指标
│   │   ├── __init__.py
│   │   └── metrics.py                            # Prometheus指标
│   │
│   └── api/                                   # 接口层
│       ├── __init__.py
│       ├── rest.py                               # FastAPI服务器
│       ├── sdk.py                                # Python SDK
│       └── mcp.py                                # MCP服务器
│
├── tests/                                     # 测试套件
│   ├── __init__.py
│   ├── conftest.py                            # pytest配置
│   ├── test_manager.py                         # 核心测试
│   ├── test_backends/                        # 后端测试
│   │   ├── test_docker.py
│   │   └── test_firecracker.py
│   ├── test_workspace/                      # Workspace测试
│   ├── test_data/                           # 数据管理测试
│   └── test_api/                            # API测试
│
├── docs/                                      # 文档
│   ├── architecture.md                        # 架构说明
│   ├── security.md                          # 安全保证
│   ├── api.md                              # API文档
│   └── performance.md                      # 性能基准
│
├── examples/                                   # 示例代码
│   ├── basic_execution.py                 # 基础执行
│   ├── model_training.py                   # 模型训练
│   └── multi_workspace.py                # 多workspace管理
│
├── scripts/                                   # 实用脚本
│   ├── setup_dev_env.sh                  # 开发环境设置
│   └── benchmark.py                         # 性能测试
│
└── deployment/                                # 部署配置
    ├── docker/
    │   └── docker-compose.yml             # 本地开发
    ├── kubernetes/
    │   ├── crds/                          # Custom Resource Definitions
    │   └── helm/                          # Helm Charts
    └── cloud/
        └── aws/                         # AWS部署
            └── ecs.tf                  # Terraform配置
```

---

## 四、核心模块

### 4.1 SandboxManager（核心编排器）

**职责**：
- 后端注册与路由
- 请求验证与策略决策
- 执行生命周期管理
- 资源配额管理

**接口设计**：
```python
class SandboxManager:
    """沙箱管理器 - 单一入口点"""

    def __init__(
        self,
        config: SandboxConfig = SandboxConfig()
    ):
        self.config = config
        self._backends: Dict[str, SandboxBackend] = {}
        self._router = IsolationRouter(config)

    async def execute(
        self,
        code: str,
        workspace_id: str,
        datasets: List[str] = None,
        mode: str = "safe"
        timeout_sec: int = 3600
    ) -> ExecutionResult:
        """
        核心执行方法

        流程：
        1. 验证workspace存在
        2. 准备datasets到workspace/data/
        3. 根据策略选择backend
        4. 挂载workspace到sandbox
        5. 执行代码
        6. 收集结果
        7. 写审计日志
        8. 返回ExecutionResult
        """
```

### 4.2 IsolationRouter（策略路由器）

**路由决策**：
```python
class IsolationRouter:
    """隔离级别路由器"""

    def decide_backend(
        self,
        request: ExecutionRequest,
        code_scan: CodeScanResult
    ) -> str:
        """
        决策逻辑：

        1. 如果request明确指定backend → 使用指定backend
        2. 如果code_scan.high_risk → Firecracker
        3. 如果有GPU需求 → Firecracker
        4. 如果网络访问 → Firecracker
        5. 默认 → Docker
        """

        risk_score = self._calculate_risk(
            code_scan.risk_score,
            request.security_context
        )

        if risk_score > 0.7:
            return "firecracker"
        elif risk_score > 0.3:
            return "docker"
        else:
            return "docker"
```

### 4.3 WorkspaceManager（工作区管理）

**职责**：
- Workspace生命周期管理
- 目录结构创建与清理
- 数据集准备与挂载

**接口**：
```python
class WorkspaceManager:
    """Workspace生命周期管理"""

    async def create(
        self,
        workspace_id: str,
        setup_dirs: List[str] = ["data", "models", "outputs"]
    ) -> Workspace:
        """
        创建workspace目录结构：
        /opt/workspaces/{workspace_id}/
          ├── data/
          ├── models/
          ├── outputs/
          └── .workspace/meta.json
        """

    async def prepare_datasets(
        self,
        workspace_id: str,
        dataset_names: List[str]
    ) -> None:
        """
        从中央数据仓库复制/链接数据集到workspace/data/

        实现：
        - 单租户：copy（隔离）
        - 多租户：link（共享）
        """

    def get_mount_config(
        self,
        workspace: Workspace
    ) -> MountConfig:
        """生成Docker/K8s挂载配置"""
```

---

## 五、公共接口

### 5.1 Execution类型系统

**核心类型定义**：
```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class ExecutionRequest(BaseModel):
    """代码执行请求"""

    # 基础参数
    code: str = Field(..., description="Python代码")
    workspace_id: str = Field(..., description="Workspace ID")

    # 数据准备
    datasets: List[str] = Field(
        default_factory=list,
        description="数据集名称列表（会准备到workspace/data/）"
    )
    data_mounts: Dict[str, str] = Field(
        default_factory=dict,
        description="自定义数据挂载（路径映射）"
    )

    # 执行控制
    mode: Literal["safe", "fast", "secure"] = Field(
        default="safe",
        description="执行模式（影响隔离级别选择）"
    )
    timeout_sec: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="超时时间（秒）"
    )

    # 资源限制
    memory_mb: int = Field(
        default=4096,
        ge=512,
        le=65536,
        description="内存限制（MB）"
    )
    cpu_cores: float = Field(
        default=2.0,
        ge=0.5,
        le=16.0,
        description="CPU核心数"
    )
    enable_gpu: bool = Field(
        default=False,
        description="是否启用GPU"
    )

    # 安全配置
    network_policy: Literal["disabled", "whitelist", "proxy"] = Field(
        default="disabled",
        description="网络访问策略"
    )
    network_whitelist: List[str] = Field(
        default_factory=list,
        description="网络白名单（当network_policy=whitelist时）"
    )

    # 环境变量
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="执行环境变量"
    )

class ExecutionResult(BaseModel):
    """执行结果"""

    success: bool = Field(..., description="是否成功")
    stdout: str = Field(..., description="标准输出")
    stderr: str = Field(default="", description="标准错误输出")

    # 执行详情
    exit_code: Optional[int] = Field(None, description="退出码")
    duration_ms: int = Field(..., description="执行耗时（毫秒）")

    # 产出
    artifacts: List[str] = Field(
        default_factory=list,
        description="生成的文件路径（相对于workspace）"
    )

    # 元数据
    execution_id: str = Field(..., description="执行ID")
    workspace_id: str = Field(..., description="Workspace ID")
    backend: str = Field(..., description="使用的后端")
    isolation_level: str = Field(..., description="实际隔离级别")

    # 审计信息
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据"
    )

class Workspace(BaseModel):
    """Workspace信息"""

    workspace_id: str = Field(..., description="Workspace唯一标识")
    host_path: str = Field(..., description="宿主机路径")
    guest_path: str = Field(default="/workspace", description="沙箱内路径")
    subdirs: List[str] = Field(
        default=["data", "models", "outputs"],
        description="Workspace子目录"
    )
    status: Literal["creating", "ready", "archived"] = Field(
        default="ready",
        description="Workspace状态"
    )
    created_at: str = Field(..., description="创建时间（ISO 8601）")
    last_used_at: Optional[str] = Field(None, description="最后使用时间")
```

### 5.2 数据管理类型

```python
class DatasetInfo(BaseModel):
    """数据集信息"""

    name: str = Field(..., description="数据集名称")
    source_path: str = Field(..., description="数据集源路径（中央仓库）")
    size_mb: float = Field(..., ge=0, description="数据集大小（MB）")
    format: Literal["csv", "parquet", "json", "excel", "feather"] = Field(
        ...,
        description="数据格式"
    )
    description: Optional[str] = Field(None, description="数据集描述")
    tags: List[str] = Field(
        default_factory=list,
        description="标签（用于分类和搜索）"
    )
    registered_at: str = Field(..., description="注册时间（ISO 8601）")

class MountConfig(BaseModel):
    """挂载配置"""

    workspace_id: str = Field(..., description="Workspace ID")
    workspace_host_path: str = Field(..., description="宿主机workspace路径")
    workspace_guest_path: str = Field(
        default="/workspace",
        description="沙箱内挂载点"
    )

    # Docker卷配置
    docker_volume_config: Optional[DockerVolumeConfig] = None

    # 数据集准备
    prepared_datasets: List[PreparedDataset] = Field(
        default_factory=list,
        description="已准备的数据集"
    )

class PreparedDataset(BaseModel):
    """已准备的数据集"""

    name: str = Field(..., description="数据集名称")
    source_path: str = Field(..., description="源文件路径")
    workspace_path: str = Field(..., description="workspace内路径")
    access_path: str = Field(..., description="代码访问路径（相对）")
    size_mb: float = Field(..., description="大小（MB）")
    strategy: Literal["copy", "link"] = Field(
        default="copy",
        description="准备策略"
    )
```

---

## 六、安全设计

### 6.1 多层防护

**安全层次**：
```
┌─────────────────────────────────────────────────┐
│              应用层防护                      │
├─────────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────────┐    │
│  │     代码扫描层              │    │
│  │  - AST静态分析                 │    │
│  │  - 危险模式匹配               │    │
│  │  - 风险评分（0-1）             │    │
│  └────────────────────────────────────────┘    │
│            ↓                                   │
│  ┌────────────────────────────────────────┐    │
│  │     策略引擎层                │    │
│  │  - 隔离级别路由               │    │
│  │  - 资源限制计算               │    │
│  └────────────────────────────────────────┘    │
│            ↓                                   │
│  ┌────────────────────────────────────────┐    │
│  │     后端隔离层                │    │
│  │  - Linux namespaces               │    │
│  │  - cgroups资源限制              │    │
│  │  - seccomp过滤器                │    │
│  │  - 网络隔离                   │    │
│  │  - 文件系统隔离               │    │
│  └────────────────────────────────────────┘    │
│            ↓                                   │
│  ┌────────────────────────────────────────┐    │
│  │     基础设施层               │    │
│  │  - 主机资源管理               │    │
│  │  - 镜像扫描                   │    │
│  │  - 容器逃逸检测               │    │
│  └────────────────────────────────────────┘    │
│                                               │
└─────────────────────────────────────────────────┘
```

### 6.2 代码扫描规则

**危险模式库**：
```python
DANGEROUS_PATTERNS = {
    # 文件操作
    "file_write": r"\b(os\.remove|os\.rmdir|shutil\.rmtree)\s*\(",

    # 网络操作
    "network": r"\b(socket\.|urllib\.|requests\.|http\.client)\s*\(",

    # 子进程
    "subprocess": r"\b(subprocess\.|Popen)\s*\(",

    # 动态执行
    "dynamic_exec": r"\b(exec|eval|compile|__import__)\s*\(",

    # 系统操作
    "system": r"\b(os\.system|sys\.exit)\s*\(",
}

RISK_WEIGHTS = {
    "file_write": 0.8,      # 高危
    "network": 0.7,          # 中高
    "subprocess": 0.6,       # 中
    "dynamic_exec": 0.9,    # 高
    "system": 0.95,          # 极高
}
```

**扫描器实现**：
```python
class CodeScanner:
    """代码静态分析器"""

    def scan(self, code: str) -> CodeScanResult:
        """
        扫描流程：
        1. AST解析代码
        2. 模式匹配检测
        3. 风险评分计算
        4. 生成扫描报告
        """

        tree = ast.parse(code)
        issues = []

        for node in ast.walk(tree):
            # 检测导入
            if isinstance(node, ast.Import):
                module = node.module if isinstance(node, ast.ImportFrom) else node.names[0]
                if module in DANGEROUS_MODULES:
                    issues.append({
                        "type": "dangerous_import",
                        "module": module,
                        "line": node.lineno,
                        "severity": "high"
                    })

            # 检测函数调用
            if isinstance(node, ast.Call):
                func_name = self._get_full_name(node.func)
                for pattern, weight in DANGEROUS_PATTERNS.items():
                    if re.search(pattern, func_name):
                        issues.append({
                            "type": "dangerous_call",
                            "function": func_name,
                            "line": node.lineno,
                            "severity": "high",
                            "weight": weight
                        })

        risk_score = self._calculate_risk_score(issues)

        return CodeScanResult(
            is_safe=risk_score < 0.3,
            risk_score=risk_score,
            issues=issues,
            recommended_isolation=self._get_isolation(risk_score)
        )

    def _calculate_risk_score(self, issues: List) -> float:
        """计算0-1之间的风险分数"""
        if not issues:
            return 0.0

        total_weight = sum(
            issue.get("weight", 0.5)
            for issue in issues
        )

        # 归一化到[0, 1]
        return min(total_weight / 3.0, 1.0)

class CodeScanResult(BaseModel):
    """代码扫描结果"""

    is_safe: bool = Field(..., description="是否安全")
    risk_score: float = Field(..., ge=0, le=1, description="风险分数（0-1）")
    issues: List[CodeIssue] = Field(
        default_factory=list,
        description="发现的安全问题"
    )
    recommended_isolation: str = Field(
        ...,
        description="建议的隔离级别"
    )

class CodeIssue(BaseModel):
    """代码问题"""
    type: str = Field(..., description="问题类型")
    line: int = Field(..., ge=1, description="行号")
    severity: str = Field(..., description="严重程度：low/medium/high/critical")
    weight: float = Field(default=0.5, description="风险权重")
    function: Optional[str] = Field(None, description="相关函数")
    module: Optional[str] = Field(None, description="相关模块")
```

### 6.3 资源限制

**cgroups配置**：
```python
class ResourceLimiter:
    """资源限制器"""

    @staticmethod
    def create_cgroup_config(
        memory_mb: int,
        cpu_cores: float,
        timeout_sec: int
    ) -> str:
        """生成cgroup配置"""

        return f"""
# Memory limit: {memory_mb}M
memory.limit_in_bytes={memory_mb * 1024 * 1024 * 1024}

# CPU limit
cpu.cfs_quota_us={cpu_cores * 100000}
cpu.cfs_period_us=100000

# Time limit (cpu time)
cpu.max={timeout_sec}
"""

    @staticmethod
    def create_docker_limits(
        memory_mb: int,
        cpu_cores: float
    ) -> Dict:
        """Docker资源限制配置"""

        return {
            "mem_limit": f"{memory_mb}m",
            "cpu_quota": f"{cpu_cores * 1e6}",
            "cpu_period": 100000,
            "pids_limit": 100,  # 限制进程数
        }
```

---

## 七、API设计

### 7.1 REST API（v1）

**核心端点**：
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="ds-sandbox API",
    version="1.0.0",
    description="General-purpose AI code execution sandbox"
)

# ========== Workspace管理 ==========

@app.post("/v1/workspaces", response_model=Workspace, status_code=201)
async def create_workspace(request: CreateWorkspaceRequest):
    """创建新workspace"""
    workspace = await workspace_manager.create(
        workspace_id=request.workspace_id,
        setup_dirs=request.setup_dirs
    )
    return workspace

@app.get("/v1/workspaces/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: str):
    """获取workspace信息"""
    return await workspace_manager.get(workspace_id)

@app.delete("/v1/workspaces/{workspace_id}", status_code=204)
async def delete_workspace(workspace_id: str):
    """删除workspace及其数据"""
    await workspace_manager.delete(workspace_id)

# ========== 数据集管理 ==========

@app.post("/v1/workspaces/{workspace_id}/datasets", status_code=200)
async def prepare_datasets(
    workspace_id: str,
    request: PrepareDatasetsRequest
):
    """准备数据集到workspace/data/"""
    await workspace_manager.prepare_datasets(
        workspace_id=workspace_id,
        datasets=request.datasets
    )
    return {"status": "prepared"}

@app.get("/v1/workspaces/{workspace_id}/datasets", response_model=List[DatasetInfo])
async def list_available_datasets():
    """列出所有可用数据集"""
    return await dataset_registry.list_all()

# ========== 代码执行（核心功能）==========

@app.post("/v1/workspaces/{workspace_id}/run",
         response_model=ExecutionInfo,
         status_code=201)
async def execute_code(
    workspace_id: str,
    request: ExecutionRequest
):
    """
    统一执行入口（推荐使用）

    流程：
    1. 验证workspace存在
    2. 扫描代码（如果启用）
    3. 决定隔离级别
    4. 准备数据集
    5. 挂载workspace
    6. 执行代码
    7. 返回ExecutionInfo（包含execution_id）
    """

    execution = await sandbox_manager.execute(
        code=request.code,
        workspace_id=workspace_id,
        datasets=request.datasets,
        mode=request.mode,
        timeout_sec=request.timeout_sec,
        env_vars=request.env_vars
    )

    return ExecutionInfo(
        execution_id=execution.execution_id,
        workspace_id=workspace_id,
        status="running"
    )

@app.get("/v1/workspaces/{workspace_id}/runs/{execution_id}",
         response_model=ExecutionStatus)
async def get_execution_status(
    workspace_id: str,
    execution_id: str
):
    """查询执行状态"""
    return await execution_tracker.get_status(execution_id)

@app.post("/v1/workspaces/{workspace_id}/runs/{execution_id}/stop",
         status_code=200)
async def stop_execution(
    workspace_id: str,
    execution_id: str
):
    """停止执行中的任务"""
    await execution_tracker.stop(execution_id)
    return {"status": "stopped"}

@app.get("/v1/workspaces/{workspace_id}/runs/{execution_id}/logs",
          response_model=ExecutionLogs)
async def get_execution_logs(
    workspace_id: str,
    execution_id: str,
    offset: int = 0,
    limit: int = 1000
):
    """获取执行日志（流式）"""
    return await execution_tracker.get_logs(
        execution_id,
        offset,
        limit
    )

# ========== 系统管理 ==========

@app.get("/v1/health", response_model=HealthStatus)
async def health_check():
    """系统健康检查"""
    backends = await backend_registry.health_check()
    return HealthStatus(
        status="healthy" if all(b.status == "ready" for b in backends) else "degraded",
        backends=backends,
        version="1.0.0"
    )

@app.get("/v1/metrics", response_model=SystemMetrics)
async def get_metrics():
    """系统指标"""
    return await metrics_collector.get_current_metrics()

# ========== 错误处理 ==========

class SandboxErrorResponse(BaseModel):
    """统一错误响应"""

    error_code: str = Field(..., description="错误代码（SBX-XXX）")
    message: str = Field(..., description="用户友好的错误描述")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外错误详情"
    )
    request_id: str = Field(..., description="请求追踪ID")
    timestamp: str = Field(..., description="错误时间（ISO 8601）")

# 错误码定义
class ErrorCode:
    """标准错误码"""
    WSP_NOT_FOUND = "SBX_WSP_001"          # Workspace不存在
    WSP_INVALID = "SBX_WSP_002"             # Workspace状态无效
    DAT_NOT_FOUND = "SBX_DAT_001"           # 数据集不存在
    DAT_NOT_PREPARED = "SBX_DAT_002"        # 数据集未准备
    EXEC_TIMEOUT = "SBX_EXEC_001"           # 执行超时
    EXEC_FAILED = "SBX_EXEC_002"             # 执行失败
    RESOURCE_LIMIT = "SBX_RES_001"          # 资源限制
    SEC_SCAN_FAILED = "SBX_SEC_001"         # 安全扫描失败
    BACKEND_UNAVAILABLE = "SBX_BAK_001"    # 后端不可用

@app.exception_handler(SandboxError)
async def sandbox_error_handler(request: Request, exc: SandboxError):
    """全局异常处理"""
    error_mapping = {
        WorkspaceNotFoundError: 404,
        DatasetNotFoundError: 400,
        DatasetNotPreparedError: 400,
        ExecutionTimeoutError: 408,
        ResourceLimitError: 413,
    }

    status_code = error_mapping.get(type(exc), 500)

    return JSONResponse(
        status_code=status_code,
        content=SandboxErrorResponse(
            error_code=exc.error_code,
            message=str(exc),
            details=exc.details if hasattr(exc, 'details') else {},
            request_id=generate_request_id()
        ).model_dump()
    )
```

### 7.2 API版本控制

**版本策略**：
```
URL格式：/v1/{resource}
响应头：X-API-Version: 1.0.0

破坏性变更：主版本号递增
向后兼容：小版本递增
Beta标记：v1.0.0-beta.1
```

---

## 八、SDK设计

### 8.1 Python SDK

**核心类**：
```python
from typing import Optional, List
import aiohttp

class SandboxSDK:
    """ds-sandbox Python SDK"""

    def __init__(
        self,
        api_endpoint: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.endpoint = api_endpoint
        self.session = aiohttp.ClientSession()
        self.api_key = api_key

    # ========== Workspace管理 ==========

    async def create_workspace(
        self,
        workspace_id: str,
        setup_dirs: List[str] = ["data", "models"]
    ) -> Workspace:
        """创建workspace"""
        async with self.session.post(
            f"{self.endpoint}/v1/workspaces",
            json={"workspace_id": workspace_id, "setup_dirs": setup_dirs},
            headers=self._headers()
        ) as resp:
            data = await resp.json()
            return Workspace(**data)

    async def prepare_datasets(
        self,
        workspace_id: str,
        datasets: List[str]
    ) -> None:
        """准备数据集"""
        async with self.session.post(
            f"{self.endpoint}/v1/workspaces/{workspace_id}/datasets",
            json={"datasets": datasets},
            headers=self._headers()
        ) as resp:
            if resp.status != 200:
                raise SandboxError.from_response(resp)

    # ========== 代码执行（主要功能）==========

    async def execute(
        self,
        workspace_id: str,
        code: str,
        mode: str = "safe",
        timeout_sec: int = 3600,
        datasets: List[str] = None,
        env_vars: dict = None
    ) -> ExecutionResult:
        """
        执行代码（同步或异步）

        Args:
            workspace_id: Workspace ID
            code: Python代码
            mode: safe/fast/secure
            timeout_sec: 超时时间
            datasets: 数据集列表
            env_vars: 环境变量

        Returns:
            ExecutionResult对象
        """

        request_data = {
            "code": code,
            "mode": mode,
            "timeout_sec": timeout_sec,
            "datasets": datasets or [],
            "env_vars": env_vars or {}
        }

        async with self.session.post(
            f"{self.endpoint}/v1/workspaces/{workspace_id}/run",
            json=request_data,
            headers=self._headers()
        ) as resp:
            if resp.status != 201:
                raise SandboxError.from_response(resp)

            data = await resp.json()
            execution_id = data["execution_id"]

            # 等待执行完成（轮询或一次性等待）
            result = await self._wait_for_completion(
                execution_id,
                timeout_sec=timeout_sec
            )

            return result

    async def get_execution_status(
        self,
        workspace_id: str,
        execution_id: str
    ) -> ExecutionStatus:
        """查询执行状态"""
        async with self.session.get(
            f"{self.endpoint}/v1/workspaces/{workspace_id}/runs/{execution_id}",
            headers=self._headers()
        ) as resp:
            if resp.status != 200:
                raise SandboxError.from_response(resp)
            return ExecutionStatus(**await resp.json())

    async def stop_execution(
        self,
        workspace_id: str,
        execution_id: str
    ) -> None:
        """停止执行"""
        async with self.session.post(
            f"{self.endpoint}/v1/workspaces/{workspace_id}/runs/{execution_id}/stop",
            headers=self._headers()
        ) as resp:
            if resp.status != 200:
                raise SandboxError.from_response(resp)

    # ========== 辅助方法 ==========

    async def list_workspaces(self) -> List[Workspace]:
        """列出所有workspace"""
        async with self.session.get(
            f"{self.endpoint}/v1/workspaces",
            headers=self._headers()
        ) as resp:
            return Workspace(**(await resp.json())

    async def delete_workspace(self, workspace_id: str) -> None:
        """删除workspace"""
        async with self.session.delete(
            f"{self.endpoint}/v1/workspaces/{workspace_id}",
            headers=self._headers()
        ) as resp:
            if resp.status != 204:
                raise SandboxError.from_response(resp)

    def _headers(self) -> dict:
        """生成请求头"""
        headers = {
            "Content-Type": "application/json",
            "X-API-Version": "1.0.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _wait_for_completion(
        self,
        execution_id: str,
        timeout_sec: int
    ) -> ExecutionResult:
        """等待执行完成"""
        start_time = time.time()

        while time.time() - start_time < timeout_sec:
            status = await self.get_execution_status(
                execution_id.split('-')[0],  # 去掉前缀
                execution_id
            )

            if status.status in ["completed", "failed"]:
                result = await self.get_execution_result(execution_id)
                return result

            await asyncio.sleep(0.5)  # 轮询间隔

        raise ExecutionTimeoutError(f"Execution {execution_id} timeout")

class ExecutionStatus(BaseModel):
    """执行状态"""
    execution_id: str
    workspace_id: str
    status: Literal["queued", "running", "completed", "failed", "stopped"]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[ExecutionResult] = None
```

### 8.2 使用示例

```python
# examples/basic_usage.py

import asyncio
from ds_sandbox import SandboxSDK

async def main():
    # 初始化SDK
    sdk = SandboxSDK(
        api_endpoint="http://localhost:8000"
    )

    try:
        # 1. 创建workspace
        workspace = await sdk.create_workspace(
            workspace_id="demo-exp-001",
            setup_dirs=["data", "models", "outputs"]
        )
        print(f"✓ Workspace created: {workspace.workspace_id}")

        # 2. 准备数据集
        await sdk.prepare_datasets(
            workspace_id="demo-exp-001",
            datasets=["bike-sharing-demand", "titanic"]
        )
        print("✓ Datasets prepared")

        # 3. 执行代码
        result = await sdk.execute(
            workspace_id="demo-exp-001",
            code="""
import pandas as pd
import os

# 查看workspace结构
print(f"Working directory: {os.getcwd()}")
print(f"Contents: {os.listdir('.')}")

# 读取数据（相对路径）
df_bike = pd.read_csv('data/bike-sharing-demand/train.csv')
print(f"Bike dataset: {df_bike.shape}")

df_titanic = pd.read_csv('data/titanic/train.csv')
print(f"Titanic dataset: {df_titanic.shape}")

# 简单分析
print(f"\\n=== Basic Statistics ===")
print(f"Bike rows: {len(df_bike)}, columns: {list(df_bike.columns)}")
print(f"Titanic rows: {len(df_titanic)}, columns: {list(df_titanic.columns)}")

# 保存模型到models/（相对路径）
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

model = RandomForestClassifier(n_estimators=10, max_depth=5)
X = df_bike[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp_1', 'atemp_2', 'atemp_3', 'atemp_4', 'humidity', 'windspeed']]
y = df_bike['count'] > df_bike['count'].median()
model.fit(X, y)

model_path = 'models/rf_bike.pkl'
dump(model, model_path)
print(f"\\nModel saved to: {model_path}")

# 验证
import os
print(f"\\nFiles in models/: {os.listdir('models/')}")
""",
            mode="fast",
            timeout_sec=600
        )

        # 4. 检查结果
        if result.success:
            print("✓ Execution succeeded")
            print(f"  Output: {result.stdout[:200]}...")
            if result.artifacts:
                print(f"  Artifacts: {result.artifacts}")
            print(f"  Duration: {result.duration_ms}ms")
        else:
            print(f"✗ Execution failed")
            print(f"  Error: {result.stderr}")

    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 九、数据管理

### 9.1 Workspace数据流

```
中央数据仓库              Workspace（执行时）
/opt/datasets/          /opt/workspaces/{id}/
  ├── bike-sharing/         └── data/
  ├── titanic/               └── models/
  └── housing/             └── outputs/

准备阶段（execute前）：
1. validate datasets
2. copy/link to /opt/workspaces/{id}/data/
3. verify integrity
4. record metadata

执行阶段：
1. mount /opt/workspaces/{id} → /workspace
2. code运行在 /workspace
3. 访问 data/{dataset}/file.csv
4. 保存到 models/{name}.pkl
```

### 9.2 数据集注册

```python
class DatasetRegistry:
    """数据集注册表"""

    def __init__(self, registry_path: str = "/opt/datasets"):
        self.registry_path = Path(registry_path)
        self._index_file = self.registry_path / ".index.json"

    async def register(
        self,
        name: str,
        source_path: str,
        format: str,
        description: str = None,
        tags: List[str] = None
    ) -> DatasetInfo:
        """
        注册新数据集

        流程：
        1. 验证source_path存在
        2. 计算size和checksum
        3. 更新索引文件
        4. 可选：创建符号链接加速
        """

        # 读取现有索引
        index = self._load_index()

        # 检查重复
        if name in index:
            raise DatasetAlreadyExistsError(name)

        # 收集元数据
        metadata = {
            "size_mb": self._calculate_size(source_path),
            "format": format,
            "checksum": self._checksum(source_path),
            "registered_at": datetime.utcnow().isoformat(),
            "description": description,
            "tags": tags or []
        }

        # 更新索引
        index[name] = {
            "source_path": str(source_path),
            "metadata": metadata
        }

        self._save_index(index)

        return DatasetInfo(
            name=name,
            source_path=source_path,
            **metadata
        )

    async def get(self, name: str) -> DatasetInfo:
        """获取数据集信息"""
        index = self._load_index()
        if name not in index:
            raise DatasetNotFoundError(name)
        return DatasetInfo(
            name=name,
            **index[name]
        )

    def _load_index(self) -> dict:
        """加载索引文件"""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self, index: dict):
        """保存索引文件"""
        self._index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_file, 'w') as f:
            json.dump(index, f, indent=2)
```

---

## 十、测试策略

### 10.1 测试金字塔

```
                /\
               /  \
              /    \ E2E Tests
             /      \______单元测试（核心模块）
            /               \     /  \
            /               \     \    \集成测试（真实环境）
           /               \     /    \
          /                \     /______性能与压力测试
         /                \    /
        /__________________________\
```

### 10.2 单元测试覆盖

```python
# tests/test_manager.py

import pytest
from ds_sandbox import SandboxManager
from ds_sandbox.types import SandboxConfig

@pytest.fixture
async def sandbox_manager():
    """测试用SandboxManager实例"""
    config = SandboxConfig(
        default_backend="docker",  # 测试用Docker后端
        workspace_base_dir="/tmp/test_workspaces"
    )
    return SandboxManager(config=config)

@pytest.mark.asyncio
async def test_create_workspace(sandbox_manager):
    """测试workspace创建"""
    workspace = await sandbox_manager.create_workspace("test-ws-001")

    assert workspace.workspace_id == "test-ws-001"
    assert workspace.host_path.exists()
    assert "data" in workspace.subdirs
    assert "models" in workspace.subdirs
    assert "outputs" in workspace.subdirs

@pytest.mark.asyncio
async def test_prepare_datasets(sandbox_manager):
    """测试数据集准备"""
    await sandbox_manager.create_workspace("test-ws-002")

    await sandbox_manager.prepare_datasets(
        workspace_id="test-ws-002",
        datasets=["test-dataset-1", "test-dataset-2"]
    )

    workspace = await sandbox_manager.get("test-ws-002")
    data_dir = workspace.host_path / "data"

    assert (data_dir / "test-dataset-1").exists()
    assert (data_dir / "test-dataset-2").exists()

@pytest.mark.asyncio
async def test_code_execution(sandbox_manager):
    """测试代码执行"""
    await sandbox_manager.create_workspace("test-ws-003")

    result = await sandbox_manager.execute(
        workspace_id="test-ws-003",
        code="print('Hello, sandbox!')",
        mode="fast"
    )

    assert result.success is True
    assert "Hello, sandbox!" in result.stdout
    assert result.execution_id is not None

@pytest.mark.asyncio
async def test_resource_limits(sandbox_manager):
    """测试资源限制"""
    await sandbox_manager.create_workspace("test-ws-004")

    with pytest.raises(TimeoutError):
        await sandbox_manager.execute(
            workspace_id="test-ws-004",
            code="import time; time.sleep(10)",
            timeout_sec=2  # 2秒超时
        )

@pytest.mark.asyncio
async def test_code_scanning(sandbox_manager):
    """测试代码扫描"""
    from ds_sandbox.security import CodeScanner

    scanner = CodeScanner()

    # 安全代码
    safe_result = scanner.scan("import pandas as pd\ndf = pd.DataFrame()")
    assert safe_result.is_safe is True
    assert safe_result.risk_score < 0.1

    # 危险代码
    dangerous_result = scanner.scan("import os; os.system('rm -rf /')")
    assert dangerous_result.is_safe is False
    assert dangerous_result.risk_score > 0.7
```

### 10.3 集成测试

```python
# tests/integration/test_docker_backend.py

import pytest
import asyncio
from ds_sandbox.backends.docker import DockerSandbox

@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_execution():
    """测试Docker后端实际执行"""
    backend = DockerSandbox()

    workspace = DockerSandbox.create_test_workspace("integration-test")

    result = await backend.execute(
        workspace=workspace,
        code="""
import pandas as pd
df = pd.read_csv('data/test.csv')
print(df.head())
""",
        timeout_sec=30
    )

    assert result.success
    assert "test.csv" in result.stdout

@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_isolation():
    """测试Docker隔离性"""
    backend = DockerSandbox()

    # 尝试访问宿主机文件（应该失败）
    result = await backend.execute(
        workspace=backend.create_test_workspace("isolation-test"),
        code="""
# 尝试读取宿主机文件
try:
    with open('/etc/passwd', 'r') as f:
        print(f'Content: {f.read()[:100]}')
except Exception as e:
    print(f'Failed: {e}')
"""
    )

    # 应该失败
    assert result.success is False
    assert "Permission denied" in result.stderr or "Operation not permitted" in result.stderr
```

---

## 十一、里程碑

### Phase 0: 项目�（1周）

**目标**：
- ✅ 项目结构搭建
- ✅ 配置文件就绪
- ✅ 基础测试框架

**交付物**：
```
ds-sandbox/
├── pyproject.toml        ✓
├── README.md              ✓
├── LICENSE                ✓
├── src/ds_sandbox/       ✓
│   ├── __init__.py      ✓
│   ├── types.py         ✓
│   └── errors.py        ✓
└── tests/                 ✓
    ├── conftest.py          ✓
    └── test_manager.py    ✓
```

### Phase 1: 核心MVP（4-6周）

**目标**：
- ✅ Docker后端实现
- ✅ Workspace管理实现
- ✅ 基础REST API
- ✅ 数据集注册与准备
- ✅ 代码扫描与策略路由
- ✅ Python SDK
- ✅ 单元测试覆盖率>80%

**验收标准**：
```bash
# 功能验证
✓ Docker backend可执行代码
✓ Workspace可创建和准备数据
✓ REST API /v1/workspaces/{id}/run 可用
✓ Python SDK可异步执行代码
✓ 数据集可准备到workspace/data/
✓ 代码扫描可检测危险操作

# 性能基准
✓ Docker启动时间: <500ms (P50)
✓ 执行吞吐: >100 exec/min (单机)
✓ 内存开销: <50MB (空载)
```

### Phase 2: 安全隔离（6-8周）

**目标**：
- ✅ Firecracker后端实现
- ✅ Kata Containers后端实现
- ✅ 完整安全策略（网络、资源）
- ✅ 审计日志系统
- ✅ 性能监控指标

**验收标准**：
```bash
# 功能验证
✓ Firecracker backend可执行代码
✓ 隔离级别自动路由工作
✓ 网络策略（disabled/whitelist）生效
✓ 资源限制（内存/CPU）生效
✓ 审计日志记录每次执行

# 安全验证
✓ 容器逃逸防护
✓ 文件系统隔离（独立rootfs）
✓ 网络隔离（独立netns）
✓ 进程隔离（独立pidns）
```

### Phase 3: K8s集成（4-6周）

**目标**：
- ✅ K8s CRD定义
- ✅ Helm Charts
- ✅ Operator实现（kopf）
- ✅ 持久化存储（PVC）

**验收标准**：
```bash
# 功能验证
✓ K8s Sandbox CRD可创建
✓ Helm安装可部署sandbox
✓ Operator自动管理sandbox生命周期
✓ PVC持久化工作正常
✓ 多租户隔离有效

# K8s验证
kubectl get sandbox -n test-001  ✓
kubectl describe workspace test-001     ✓
kubectl logs -f sandbox/test-001      ✓
```

---

## 十二、技术选型

### 12.1 核心依赖

```toml
[project.dependencies]
# 核心框架
fastapi = "^0.100.0"          # Web框架
pydantic = "^2.0"               # 数据验证
pydantic-settings = "^2.0"       # 配置管理
aiofiles = "^23.0"              # 异步文件操作

# Jupyter/笔记本执行
nbclient = "^0.10.0"             # Notebook执行引擎
nbformat = "^5.0.0"              # Notebook格式

# Docker集成
docker = "^7.0.0"                # Docker SDK

# 开发工具（开发依赖）
pytest = "^7.0.0"                # 测试框架
pytest-asyncio = "^0.21.0"      # 异步测试
pytest-cov = "^4.0.0"             # 覆盖率
ruff = "^0.1.0"                   # 代码格式化
mypy = "^1.0.0"                   # 类型检查
black = "^23.0.0"                  # 代码格式化（可选）

# 文档工具
mkdocs = "^1.5.0"                 # 文档生成
mkdocs-material = "^9.0.0"         # 主题

# 可选依赖（按后端）
firecracker-go = {version = ">=1.0.0", optional = true}  # Firecracker
```

### 12.2 Docker配置

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 安装ds-sandbox
RUN pip install .

# 默认配置
ENV SANDBOX_DEFAULT_BACKEND=docker
ENV SANDBOX_WORKSPACE_BASE=/opt/workspaces
ENV SANDBOX_DATASET_DIR=/opt/datasets

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/v1/health || exit 1

# 运行API服务器
CMD ["uvicorn", "ds_sandbox.api.rest:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--log-level", "info"]
```

### 12.3 运行时配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  ds-sandbox-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SANDBOX_DEFAULT_BACKEND=docker
      - SANDBOX_WORKSPACE_BASE=/opt/workspaces
      - SANDBOX_DATASET_DIR=./test_datasets
    volumes:
      - ./data:/opt/datasets
      - ./workspaces:/opt/workspaces
```

---

## 附录A：快速开始指南

### 5分钟体验ds-sandbox

```bash
# 1. 安装ds-sandbox
pip install ds-sandbox

# 2. 启动API服务器（默认Docker后端）
ds-sandbox-api &

# 3. 创建workspace并执行代码
curl -X POST http://localhost:8000/v1/workspaces \
  -H "Content-Type: application/json" \
  -d '{"workspace_id": "demo-001"}'

# 4. 执行Python代码
curl -X POST http://localhost:8000/v1/workspaces/demo-001/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import pandas as pd; print(pd.__version__)",
    "mode": "fast"
  }'

# 预期响应：
# {
#   "execution_id": "exec-123456",
#   "workspace_id": "demo-001",
#   "status": "running"
# }
```

---

## 附录B：与DSLighting集成

### DSLighting使用ds-sandbox

```python
# dslighting/dslighting/sandbox/adapter.py

"""
DSLighting Sandbox适配器
桥接DSLighting的workspace服务到ds-sandbox
"""

from ds_sandbox import SandboxSDK
from dslighting.services.workspace import WorkspaceService

class SandboxService:
    """DSLighting Sandbox服务（适配器模式）"""

    def __init__(self, workspace: WorkspaceService):
        """
        Args:
            workspace: DSLighting的workspace服务
        """
        self.workspace = workspace

        # 初始化ds-sandbox SDK
        self.sdk = SandboxSDK(
            api_endpoint="http://localhost:8000"  # ds-sandbox API
        )

        # 将DSLighting workspace映射到ds-sandbox workspace
        self._ensure_sandbox_workspace()

    def _ensure_sandbox_workspace(self) -> None:
        """确保ds-sandbox中有对应workspace"""
        # 通过SDK创建workspace（如果不存在）
        if not hasattr(self, '_sandbox_workspace_created'):
            from ds_sandbox import Workspace as SandboxWorkspace

            sandbox_ws = SandboxWorkspace(
                workspace_id=self.workspace.workspace_id,
                host_path="/opt/sandbox-workspaces",  # ds-sandbox路径
                subdirs=["data", "models", "outputs", "code"]
            )

            # 调用ds-sandbox API创建workspace
            # （这里会实际创建目录结构）
            self._sandbox_workspace_created = True

    async def run_script(
        self,
        script_code: str,
        timeout: int = 600
    ) -> ExecutionResult:
        """
        执行Python脚本（向后兼容）

        流程：
        1. 通过ds-sandbox执行代码
        2. 处理结果和错误
        """
        try:
            result = await self.sdk.execute(
                workspace_id=self.workspace.workspace_id,
                code=script_code,
                mode="safe"
            )

            # 转换结果格式以匹配DSLighting期望
            return ExecutionResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                exc_type=result.exc_type if not result.success else None,
                metadata=result.metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                stderr=str(e),
                exc_type=type(e).__name__
            )

    async def notebook_executor(self, timeout: int):
        """Notebook执行器（向后兼容）"""
        from ds_sandbox.api import NotebookExecutor as SandboxNotebookExecutor

        executor = SandboxNotebookExecutor(
            workspace_id=self.workspace.workspace_id,
            api_endpoint="http://localhost:8000"
        )

        return await executor.start()
```

**配置集成**：
```toml
# dslighting/pyproject.toml

[project.dependencies]
# 添加ds-sandbox依赖
ds-sandbox = "^1.0.0"  # 版本要求

[project.optional-dependencies]
# 开发依赖会自动安装
```

---

## 附录A：5分钟快速验证

### 步骤1：克隆并安装（2分钟）

```bash
git clone https://github.com/usail-hkust/ds-sandbox.git
cd ds-sandbox
pip install -e .
```

### 步骤2：创建workspace并执行代码（3分钟）

```bash
# 启动API服务器（默认Docker）
ds-sandbox-api &

# 在另一个终端执行代码
python - << 'EOF'
import asyncio
from ds_sandbox import SandboxSDK

async def main():
    sdk = SandboxSDK()

    # 创建workspace
    ws = await sdk.create_workspace("quickstart", ["data", "models"])
    print(f"Workspace: {ws.host_path}")

    # 执行代码
    result = await sdk.execute(
        workspace_id="quickstart",
        code="import pandas as pd; print(pd.__version__)"
    )

    print(result.stdout)
    print(f"Execution ID: {result.execution_id}")

asyncio.run(main())
EOF
```

### 步骤3：验证结果（可选，最长5分钟）

**预期输出**：
```
✅ Workspace: /opt/workspaces/quickstart
✅ Status: ready
✅ Execution completed
Output: 2.0.3
```

---

## 📊 项目文件总览

已创建的核心文件：
```
✓ README.md                                    - 项目说明
✓ LICENSE                                      - Apache-2.0协议
✓ PROPOSAL.md                                 - 完整技术方案
✓ pyproject.toml                               - 打包配置
✓ src/ds_sandbox/                            - 源代码包
  ✓ __init__.py
  ✓ types.py                                  - 类型定义
  ✓ errors.py                                 - 异常体系
  ✓ config.py                                 - 配置模型
  ✓ manager.py                                - 核心管理器（骨架）
  ✓ backends/
    ✓ __init__.py                           - 后端基类
    ✓ docker.py                              - Docker实现（核心代码）
✓ workspace/                                 - Workspace管理（骨架）
  ✓ data/                                      - 数据管理（骨架）
  ✓ storage/                                   - 存储抽象（骨架）
  ✓ security/                                  - 安全层（骨架）
  ✓ monitoring/                                - 监控（骨架）
  ✓ api/                                      - API层（骨架）
    ✓ examples/basic_execution.py          - 使用示例
✓ setup.py                                     - 安装脚本
✓ .gitignore                                  - Git忽略规则
```

**下一步**：
1. 实现核心管理器（manager.py）- Phase 1，Week 1-2
2. 完整Docker后端实现（backends/docker.py）- Phase 1，Week 3-4
3. 实现基础API服务器（api/rest.py）- Phase 1，Week 3-4
4. 添加单元测试 - Phase 1，Week 4-6

**就绪状态**：
- ✅ 项目结构完整
- ✅ 类型系统定义清晰
- ✅ 错误体系就绪
- ✅ 配置管理实现
- ✅ Docker后端框架完整

**可立即验证**：
```bash
cd /Users/liufan/projects/share/ds-sandbox
pip install -e .
python -m pytest tests/ -v  # 运行基础测试（骨架会通过）
python examples/basic_execution.py  # 运行示例（会失败，因为没有实现）
```

---

## 结语

ds-sandbox项目定位为**通用的AI代码执行沙箱框架**，填补当前开源方案的空白。通过Workspace-First的数据管理、可插拔的后端架构和完善的API设计，为AI agent提供生产级的代码执行能力。

**核心优势**：
1. ✅ **完全独立** - 零上层依赖，可单独发布和使用
2. ✅ **Workspace原生** - 数据在workspace相对路径，符合DS习惯
3. ✅ **策略驱动** - 根据风险自动选择隔离级别
4. ✅ **生产就绪** - 审计、监控、限流齐全
5. ✅ **易于集成** - REST/SDK/MCP三种接口

**预期影响**：
- 为AI agent提供可靠的代码执行环境
- 统一的数据科学沙箱标准
- 降低多项目集成成本

---

**文档版本**: v1.0
**最后更新**: 2026-02-12
**状态**: 待审核

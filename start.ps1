[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Reload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot 'backend'
$FrontendRoot = Join-Path $ProjectRoot 'frontend'
$EnvFile = Join-Path $ProjectRoot '.env'
$VenvRoot = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'

function Import-DotEnv {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host '未找到 .env，将使用程序默认值；可复制 .env.example 后自定义。' -ForegroundColor Yellow
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) {
            continue
        }

        $separator = $trimmed.IndexOf('=')
        if ($separator -lt 1) {
            continue
        }

        $name = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

function Resolve-Python {
    if (Test-Path -LiteralPath $VenvPython) {
        return $VenvPython
    }

    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $systemPython) {
        throw '未找到 Python。请安装 Python 3.11 或更高版本后重试。'
    }

    if ($Install) {
        Write-Host '正在创建 Python 虚拟环境...' -ForegroundColor Cyan
        & $systemPython.Source -m venv $VenvRoot
        if ($LASTEXITCODE -ne 0) {
            throw '创建 Python 虚拟环境失败。'
        }
        return $VenvPython
    }

    Write-Host '未找到 .venv，将临时使用系统 Python。首次运行建议执行 .\start.ps1 -Install。' -ForegroundColor Yellow
    return $systemPython.Source
}

if (-not (Test-Path -LiteralPath $BackendRoot)) {
    throw "后端目录不存在：$BackendRoot"
}
if (-not (Test-Path -LiteralPath $FrontendRoot)) {
    throw "前端目录不存在：$FrontendRoot"
}

Import-DotEnv -Path $EnvFile
$PythonExe = Resolve-Python
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $Npm) {
    $Npm = Get-Command npm -ErrorAction SilentlyContinue
}
if (-not $Npm) {
    throw '未找到 npm。请安装 Node.js 20 或更高版本后重试。'
}

if ($Install) {
    Write-Host '正在安装后端依赖...' -ForegroundColor Cyan
    & $PythonExe -m pip install -r (Join-Path $BackendRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        throw '后端依赖安装失败。'
    }

    Write-Host '正在安装前端依赖...' -ForegroundColor Cyan
    & $Npm.Source install --prefix $FrontendRoot
    if ($LASTEXITCODE -ne 0) {
        throw '前端依赖安装失败。'
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot 'node_modules'))) {
    throw '前端依赖尚未安装，请先执行 .\start.ps1 -Install。'
}

$backendHost = if ($env:BACKEND_HOST) { $env:BACKEND_HOST } else { '127.0.0.1' }
$backendPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { '8000' }
$frontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { '5173' }

$backendArgs = @('-m', 'uvicorn', 'app.main:app', '--host', $backendHost, '--port', $backendPort)
if ($Reload) {
    $backendArgs += '--reload'
}

Write-Host "后端：http://$backendHost`:$backendPort  API 文档：http://$backendHost`:$backendPort/docs" -ForegroundColor Green
Write-Host "前端：http://127.0.0.1`:$frontendPort" -ForegroundColor Green
Write-Host '按 Ctrl+C 停止两个服务。' -ForegroundColor DarkGray

$BackendProcess = $null
$FrontendProcess = $null
try {
    $BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList $backendArgs -WorkingDirectory $BackendRoot -NoNewWindow -PassThru
    $FrontendProcess = Start-Process -FilePath $Npm.Source -ArgumentList @('run', 'dev', '--', '--port', $frontendPort) -WorkingDirectory $FrontendRoot -NoNewWindow -PassThru

    while (-not $BackendProcess.HasExited -and -not $FrontendProcess.HasExited) {
        Start-Sleep -Seconds 1
    }

    if ($BackendProcess.HasExited) {
        throw "后端进程已退出，退出码：$($BackendProcess.ExitCode)"
    }
    throw "前端进程已退出，退出码：$($FrontendProcess.ExitCode)"
}
finally {
    foreach ($process in @($BackendProcess, $FrontendProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

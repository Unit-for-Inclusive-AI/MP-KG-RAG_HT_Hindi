<#
.SYNOPSIS
  One-time Windows setup for the local MP-KG-RAG Pipeline Lab.
.DESCRIPTION
  Creates .venv-local-demo with Python 3.11, installs requirements-windows.txt,
  optionally pulls the Ollama model, and optionally builds the BGE index.
.EXAMPLE
  .\setup_windows.ps1 -PullModel -BuildIndex
#>
[CmdletBinding()]
param(
    [string]$Model = $(if ($env:MPKG_OLLAMA_MODEL) { $env:MPKG_OLLAMA_MODEL } else { "qwen3:4b" }),
    [switch]$PullModel,
    [switch]$BuildIndex
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv-local-demo\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "The Python launcher 'py' was not found. Install Python 3.11 (64-bit) from python.org first."
    }
    Write-Host "Creating .venv-local-demo with Python 3.11..."
    & py -3.11 -m venv .venv-local-demo
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment with 'py -3.11'. Is Python 3.11 installed?" }
}
Write-Host "Installing requirements-windows.txt (CPU torch, Gradio, sentence-transformers)..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed; see the output above." }

if ($PullModel) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) { throw "Ollama is not installed or not on PATH. Install it from https://ollama.com/download (or: winget install Ollama.Ollama)." }
    & $ollama.Source pull $Model
    if ($LASTEXITCODE -ne 0) { throw "ollama pull $Model failed." }
}

if ($BuildIndex) {
    $env:TOKENIZERS_PARALLELISM = "false"
    & $venvPython work\build_local_mpkg_demo_index.py
    if ($LASTEXITCODE -ne 0) { throw "Index build failed; see the output above." }
}
Write-Host "Setup complete. Launch with: .\run_demo.ps1"

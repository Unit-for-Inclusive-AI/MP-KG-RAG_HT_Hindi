<#
.SYNOPSIS
  Launch the local MP-KG-RAG Pipeline Lab (Gradio) on Windows.
.DESCRIPTION
  Checks the Ollama server and model, checks the retrieval index, then starts
  the Gradio UI. Settings can also come from MPKG_* environment variables.
.EXAMPLE
  .\run_demo.ps1
  .\run_demo.ps1 -Model qwen3:8b -Port 7861
  .\run_demo.ps1 -OllamaUrl http://192.168.1.20:11434
#>
[CmdletBinding()]
param(
    [string]$Model = $(if ($env:MPKG_OLLAMA_MODEL) { $env:MPKG_OLLAMA_MODEL } else { "qwen3:4b" }),
    [string]$OllamaUrl = $(if ($env:MPKG_OLLAMA_URL) { $env:MPKG_OLLAMA_URL } else { "http://localhost:11434" }),
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 7860,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = $(if ($env:MPKG_RETRIEVAL_DEVICE) { $env:MPKG_RETRIEVAL_DEVICE } else { "cpu" }),
    [switch]$BuildIndex,
    [switch]$Offline
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv-local-demo\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "Missing .venv-local-demo. Run .\setup_windows.ps1 first." }

$env:MPKG_GENERATOR = "ollama"
$env:MPKG_OLLAMA_MODEL = $Model
$env:MPKG_OLLAMA_URL = $OllamaUrl
$env:MPKG_RETRIEVAL_DEVICE = $Device
$env:TOKENIZERS_PARALLELISM = "false"
if ($Offline) { $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1" }

# 1. Ollama server: start the local one if it is not answering yet.
function Test-Ollama { try { Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 3 } catch { $null } }
$version = Test-Ollama
if (-not $version -and $OllamaUrl -match "localhost|127\.0\.0\.1") {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) { $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"; if (Test-Path $candidate) { $ollama = Get-Item $candidate } }
    if ($ollama) {
        Write-Host "Starting 'ollama serve'..."
        $exe = if ($ollama.Source) { $ollama.Source } else { $ollama.FullName }
        Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden
        foreach ($i in 1..20) { Start-Sleep -Milliseconds 500; $version = Test-Ollama; if ($version) { break } }
    }
}
if (-not $version) { throw "Ollama is not reachable at $OllamaUrl. Start the Ollama app (or 'ollama serve') and retry." }
Write-Host "Ollama $($version.version) at $OllamaUrl"

# 2. Model: must already be pulled; never substitute another model silently.
$tags = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 10
$names = @($tags.models | ForEach-Object { $_.name })
$wanted = if ($Model -match ":") { @($Model) } else { @($Model, "$Model`:latest") }
if (-not ($names | Where-Object { $wanted -contains $_ })) {
    throw "Model '$Model' is not installed in Ollama. Run: ollama pull $Model   (installed: $($names -join ', '))"
}

# 3. Retrieval index: build on request, otherwise require a valid one.
if ($BuildIndex) {
    $buildArgs = @("work\build_local_mpkg_demo_index.py")
    if ($Offline) { $buildArgs += "--offline" }
    & $venvPython @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "Index build failed." }
} elseif (-not (Test-Path (Join-Path $PSScriptRoot "local_mpkg_demo_index\CURRENT"))) {
    throw "No retrieval index yet. Run: .\run_demo.ps1 -BuildIndex   (one-time; several hours on a laptop CPU)"
}

# 4. Gradio UI.
Write-Host "Launching Gradio on http://$BindHost`:$Port  (model $Model, retrieval device $Device)"
& $venvPython -m local_mpkg_demo.app --project-root $PSScriptRoot --host $BindHost --port $Port
exit $LASTEXITCODE

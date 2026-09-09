param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$ConfigPath = Join-Path $ProjectRoot "config.yaml"
$ExampleConfigPath = Join-Path $ProjectRoot "config.example.yaml"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Copy-Item -LiteralPath $ExampleConfigPath -Destination $ConfigPath
    Write-Host "Created config.yaml from config.example.yaml (existing configs are never overwritten)."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating an isolated Python 3.10 environment..."
    py -3.10 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 virtual-environment creation failed (exit $LASTEXITCODE)."
    }
}

Write-Host "Installing pinned dependencies (including OpenCV contrib/ArUco)..."
& $VenvPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)." }
& $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit $LASTEXITCODE)." }

if (-not $SkipTests) {
    Write-Host "Running convention and metric tests..."
    Push-Location $ProjectRoot
    try {
        & $VenvPython -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "Tests failed (exit $LASTEXITCODE)." }
    } finally {
        Pop-Location
    }
}

Write-Host "Ready. Activate with: .\.venv\Scripts\Activate.ps1"

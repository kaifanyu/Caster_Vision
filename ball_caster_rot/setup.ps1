$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating Python 3.10 virtual environment..."
    py -3.10 -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Python virtual-environment creation failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Installing pinned dependencies..."
& $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE."
}

Write-Host "Running geometry/convention tests..."
Push-Location $ProjectRoot
try {
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "Tests failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"

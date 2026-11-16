param(
    [ValidateSet('all', 'cpu', 'cuda-any', 'cuda-derived')]
    [string]$Target = 'all'
)

$ErrorActionPreference = 'Stop'
$buildDir = Join-Path $PSScriptRoot 'build'
$tempDir = Join-Path $buildDir 'temp'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

if ($Target -in @('all', 'cpu')) {
    $compiler = Get-Command cl -ErrorAction SilentlyContinue
    if (-not $compiler) {
        throw 'cl was not found. Run this script from a Visual Studio Developer PowerShell.'
    }
    $env:TEMP = $tempDir
    $env:TMP = $tempDir
    Push-Location $buildDir
    try {
        & $compiler.Source /O2 /EHsc /std:c++17 (Join-Path $PSScriptRoot 'seedmine2d.cpp') /Fe:seedmine2d.exe
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }
}

if ($Target -in @('all', 'cuda-any', 'cuda-derived')) {
    $compiler = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $compiler) {
        throw 'nvcc was not found. Install the NVIDIA CUDA Toolkit and add nvcc to PATH.'
    }
    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        throw 'cl was not found. Run this script from a Visual Studio Developer PowerShell.'
    }
    $env:TEMP = $tempDir
    $env:TMP = $tempDir
}

if ($Target -in @('all', 'cuda-any')) {
    & $compiler.Source -O3 -std=c++20 (Join-Path $PSScriptRoot 'seedmine_cuda_any.cu') -o (Join-Path $buildDir 'seedmine_cuda_any.exe')
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }
}

if ($Target -in @('all', 'cuda-derived')) {
    & $compiler.Source -O3 -std=c++20 (Join-Path $PSScriptRoot 'minesweep_derived_bucket.cu') -o (Join-Path $buildDir 'minesweep_derived_bucket.exe')
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }
}

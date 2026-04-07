# Chay tu thu muc goc: .\scripts\docker-compose.ps1 train-softmax
param(
    [Parameter(Position = 0)]
    [string]$Command = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
switch ($Command) {
    "build" { docker compose build @Rest }
    "shell" { docker compose run --rm app bash @Rest }
    "woe" { docker compose run --rm woe-analysis @Rest }
    "simplified" { docker compose run --rm create-simplified @Rest }
    "sample" { docker compose run --rm sample-data @Rest }
    "train-softmax" { docker compose run --rm train-mlp-softmax @Rest }
    "validate-softmax" { docker compose run --rm validate-mlp-softmax @Rest }
    "test-softmax" { docker compose run --rm test-mlp-softmax @Rest }
    "train-coral" { docker compose run --rm train-mlp-coral @Rest }
    "validate-coral" { docker compose run --rm validate-mlp-coral @Rest }
    "test-coral" { docker compose run --rm test-mlp-coral @Rest }
    default {
        Write-Host "Usage: .\scripts\docker-compose.ps1 {build|shell|woe|simplified|sample|train-softmax|...}"
        exit 1
    }
}

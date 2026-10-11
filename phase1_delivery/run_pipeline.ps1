param(
    [ValidateSet('Verify','Analyze','Recompute')][string]$Stage='Verify',
    [string]$Python='',
    [string]$Run='full4096_fast'
)
$ErrorActionPreference='Stop'
if (-not $Python) {
    $taskRuntime='D:\CodexResearch\neural-microscope\.venv\Scripts\python.exe'
    $Python=if (Test-Path -LiteralPath $taskRuntime) {$taskRuntime} else {'python'}
}
function Invoke-Phase([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {throw "Pipeline step failed: $($Arguments -join ' ')"}
}
if ($Stage -eq 'Recompute') {
    if ($Run -eq 'full4096_fast') {$Run='full4096_recomputed'}
    Invoke-Phase @("$PSScriptRoot/code/phase1.py",'train','--config',"$PSScriptRoot/configs/full_fast.json",'--steps','10000','--run-name',$Run)
    Invoke-Phase @("$PSScriptRoot/code/phase1.py",'collect','--run',$Run,'--per-condition','2048','--ext-per-condition','256','--chunk','128')
    Invoke-Phase @("$PSScriptRoot/code/phase1.py",'index','--run',$Run)
}
if ($Stage -ne 'Analyze') {Invoke-Phase @("$PSScriptRoot/code/phase1.py",'verify','--run',$Run)}
if ($Stage -ne 'Verify') {Invoke-Phase @("$PSScriptRoot/code/analyze.py",'--run',$Run)}

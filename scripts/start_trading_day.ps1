param(
    [string]$Config = 'config/trading.local.yaml',
    [string]$Account = 'panda_contest',
    [string]$Python = 'python',
    [switch]$Supervise,
    [string]$LogStamp
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$settingsPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $workspace $Config }
$settingsPath = (Resolve-Path -LiteralPath $settingsPath).Path
$pythonPath = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
$entryPath = Join-Path $workspace 'run_trading_workflow.py'
if ($Account -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid account identifier' }
foreach ($item in @($settingsPath, $entryPath)) {
    if ($item.Contains('"')) { throw 'A command argument contains a quote' }
}
$logDirectory = Join-Path $workspace 'runs/trading/launcher'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stamp = if ($LogStamp) { $LogStamp } else { [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') }
if ($stamp -notmatch '^\d{8}T\d{9}Z$') { throw 'Invalid log timestamp' }
$stdoutPath = Join-Path $logDirectory "$Account-$stamp.out.log"
$stderrPath = Join-Path $logDirectory "$Account-$stamp.err.log"
$exitPath = Join-Path $logDirectory "$Account-$stamp.exit.json"
if (-not $Supervise) {
    # Keep a small observer alive after the calling terminal returns. It records
    # the actual Python exit code; it never restarts or submits on its own.
    $supervisorArgs = @('-NoProfile', '-NonInteractive', '-File', ('"' + $PSCommandPath + '"'),
        '-Config', ('"' + $settingsPath + '"'), '-Account', $Account,
        '-Python', ('"' + $pythonPath + '"'), '-Supervise', '-LogStamp', $stamp)
    $supervisor = Start-Process -FilePath (Get-Command pwsh -ErrorAction Stop).Source `
        -ArgumentList $supervisorArgs -WorkingDirectory $workspace -WindowStyle Hidden -PassThru
    $launch = [ordered]@{ pid = $supervisor.Id; role = 'supervisor'; started_at = [DateTimeOffset]::Now.ToString('o');
        stdout = $stdoutPath; stderr = $stderrPath; exit_record = $exitPath }
    $launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDirectory "$Account-$stamp.launch.json") -Encoding utf8
    $launch | ConvertTo-Json -Compress
    return
}
# Explicit UTF-8 is necessary for the artifact path printed by the child on Windows.
$arguments = @('-u', '-X', 'utf8', '-X', 'faulthandler', '-B', ('"' + $entryPath + '"'), '--config',
               ('"' + $settingsPath + '"'), 'auto', '--account', $Account)
$exitRecord = [ordered]@{ started_at = [DateTimeOffset]::Now.ToString('o'); account = $Account;
    config = $settingsPath; supervisor_pid = $PID; exit_code = $null }
try {
    $runnerProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $workspace `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $exitRecord['pid'] = $runnerProcess.Id
    $runnerProcess.WaitForExit()
    $exitRecord['exit_code'] = $runnerProcess.ExitCode
} catch {
    $exitRecord['error'] = $_.Exception.Message
} finally {
    $exitRecord['finished_at'] = [DateTimeOffset]::Now.ToString('o')
    $exitRecord | ConvertTo-Json | Set-Content -LiteralPath $exitPath -Encoding utf8
}

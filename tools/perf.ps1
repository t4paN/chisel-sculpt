<#
.SYNOPSIS
  Manual performance runs for Chisel on Windows: launch a build with frame-time
  recording on, sculpt by hand, close the window, get a summary. Compare runs.

.EXAMPLE
  .\tools\perf.ps1 gl                       # run the OpenGL build
  .\tools\perf.ps1 wgpu -Label "L8 clay"    # run the WebGPU build, tag the run
  .\tools\perf.ps1 gl -NoVsync              # uncapped: measure throughput, not 60 Hz
  .\tools\perf.ps1 gl -Open my.chisel       # open a file on launch
  .\tools\perf.ps1 compare                  # latest gl run vs latest wgpu run
  .\tools\perf.ps1 compare -Runs a,b        # two specific run folders
  .\tools\perf.ps1 list                     # list recorded runs

  Each run lands in perf-runs\<timestamp>-<backend>[-label]\ with the full log
  (log.txt), one row per frame (frames.csv) and the parsed summary (summary.txt).
#>
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('gl', 'wgpu', 'compare', 'list')]
    [string]$Mode,
    [string]$Label = '',
    [switch]$NoVsync,
    [string]$Open = '',
    [string[]]$Runs = @(),
    [string]$Config = 'Release'
)

$ErrorActionPreference = 'Stop'
$repo    = Split-Path -Parent $PSScriptRoot
$runsDir = Join-Path $repo 'perf-runs'
$exes    = @{
    gl   = Join-Path $repo "build-gl\$Config\chisel.exe"
    wgpu = Join-Path $repo "build-wgpu\$Config\chisel.exe"
}

# ---- log parsing -------------------------------------------------------------

function Read-Run([string]$dir) {
    $log = Join-Path $dir 'log.txt'
    $lines = if (Test-Path $log) { Get-Content $log } else { @() }
    $r = [ordered]@{
        Dir = $dir; Name = Split-Path $dir -Leaf
        Backend = ''; Renderer = ''; Vsync = ''
        Phases = [ordered]@{}; Switches = @(); Strokes = @(); Penups = @(); Session = ''
    }
    foreach ($l in $lines) {
        if ($l -match '^Chisel .*\(WebGPU\)') { $r.Backend = 'WebGPU' }
        elseif ($l -match '^OpenGL (.*)')     { $r.Backend = "OpenGL $($matches[1])" }
        elseif ($l -match '^Renderer: (.*)')  { $r.Renderer = $matches[1] }
        elseif ($l -match '^\[perf\] recording frame times \(vsync (\w+)\)') { $r.Vsync = $matches[1] }
        elseif ($l -match '^\[perf\] ---- frame times \(ms\), (\d+) s session') { $r.Session = "$($matches[1]) s" }
        elseif ($l -match '^\[perf\] (L\d+ \w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)') {
            $r.Phases[$matches[1]] = [pscustomobject]@{
                Frames = [int]$matches[2]; Mean = [double]$matches[3]; P50 = [double]$matches[4]
                P95 = [double]$matches[5]; P99 = [double]$matches[6]; Max = [double]$matches[7]; Fps = [double]$matches[8]
            }
        }
        elseif ($l -match '^\[multires\] switched to level (\d+) \((\d+) verts.*in ([\d.]+) ms') {
            $r.Switches += [pscustomobject]@{ Level = [int]$matches[1]; Verts = [int]$matches[2]; Ms = [double]$matches[3] }
        }
        elseif ($l -match '^\[stage\] stroke wall (\d+) ms over (\d+) dabs \(([\d.]+) ms/dab\)') {
            $r.Strokes += [pscustomobject]@{ WallMs = [double]$matches[1]; Dabs = [int]$matches[2]; MsPerDab = [double]$matches[3] }
        }
        elseif ($l -match '^\[penup\] (\d+) ms pen-up -> committed over (\d+) frames') {
            $r.Penups += [pscustomobject]@{ Ms = [double]$matches[1]; Frames = [int]$matches[2] }
        }
    }
    [pscustomobject]$r
}

function Stat($xs) {
    if (-not $xs -or $xs.Count -eq 0) { return '-' }
    $s = $xs | Sort-Object
    $med = $s[[int][math]::Floor(($s.Count - 1) / 2)]
    '{0} runs, median {1:N1}, max {2:N1}' -f $s.Count, $med, $s[-1]
}

function Format-Run($r) {
    $out = @()
    $out += "Run      : $($r.Name)"
    $out += "Backend  : $($r.Backend)   Renderer: $($r.Renderer)   vsync: $($r.Vsync)   session: $($r.Session)"
    $out += ''
    $out += 'Frame times (ms)'
    $out += ('  {0,-12} {1,7} {2,8} {3,8} {4,8} {5,8} {6,8} {7,8}' -f 'phase', 'frames', 'mean', 'p50', 'p95', 'p99', 'max', 'fps')
    foreach ($k in $r.Phases.Keys) {
        $p = $r.Phases[$k]
        $out += ('  {0,-12} {1,7} {2,8:N2} {3,8:N2} {4,8:N2} {5,8:N2} {6,8:N2} {7,8:N1}' -f $k, $p.Frames, $p.Mean, $p.P50, $p.P95, $p.P99, $p.Max, $p.Fps)
    }
    if ($r.Phases.Count -eq 0) { $out += '  (no [perf] summary in the log - did the app crash or get killed?)' }
    $out += ''
    $out += 'Level switches'
    foreach ($s in $r.Switches) { $out += ('  -> L{0,-2} {1,10:N0} verts  {2,9:N1} ms' -f $s.Level, $s.Verts, $s.Ms) }
    if ($r.Switches.Count -eq 0) { $out += '  (none)' }
    $out += ''
    $out += "Strokes  : wall ms   $(Stat ($r.Strokes | ForEach-Object WallMs))"
    $out += "           ms/dab    $(Stat ($r.Strokes | ForEach-Object MsPerDab))"
    $out += "Pen-up   : ms        $(Stat ($r.Penups  | ForEach-Object Ms))"
    $out -join "`n"
}

function Get-Runs { if (Test-Path $runsDir) { Get-ChildItem $runsDir -Directory | Sort-Object Name } else { @() } }

# ---- modes --------------------------------------------------------------------

if ($Mode -eq 'list') {
    Get-Runs | ForEach-Object {
        $r = Read-Run $_.FullName
        '{0,-48} {1,-28} phases={2,-3} strokes={3}' -f $r.Name, $r.Backend, $r.Phases.Count, $r.Strokes.Count
    }
    return
}

if ($Mode -eq 'compare') {
    $all = Get-Runs
    if ($Runs.Count -eq 2) {
        $pick = $Runs | ForEach-Object {
            $n = $_
            $hit = $all | Where-Object { $_.Name -eq $n -or $_.FullName -eq $n } | Select-Object -Last 1
            if (-not $hit) { throw "No run named '$n' under $runsDir" }
            $hit
        }
    } else {
        $pick = @(
            ($all | Where-Object Name -match '-gl($|-)'   | Select-Object -Last 1),
            ($all | Where-Object Name -match '-wgpu($|-)' | Select-Object -Last 1)
        ) | Where-Object { $_ }
        if ($pick.Count -ne 2) { throw 'Need at least one gl run and one wgpu run (or pass -Runs a,b).' }
    }
    $a = Read-Run $pick[0].FullName; $b = Read-Run $pick[1].FullName
    "A: $($a.Name)  [$($a.Backend), vsync $($a.Vsync)]"
    "B: $($b.Name)  [$($b.Backend), vsync $($b.Vsync)]"
    if ($a.Vsync -ne $b.Vsync) { "WARNING: vsync differs between runs - frame times are not comparable." }
    ''
    '{0,-12} | {1,8} {2,8} {3,8} | {4,8} {5,8} {6,8} | {7,9}' -f 'phase', 'A p50', 'A p95', 'A frm', 'B p50', 'B p95', 'B frm', 'B/A p50'
    '-' * 86
    $keys = @($a.Phases.Keys) + @($b.Phases.Keys) | Select-Object -Unique |
        Sort-Object { [int](($_ -split ' ')[0].Substring(1)) }, { $_ }
    foreach ($k in $keys) {
        $pa = $a.Phases[$k]; $pb = $b.Phases[$k]
        $ratio = if ($pa -and $pb -and $pa.P50 -gt 0) { '{0:N2}x' -f ($pb.P50 / $pa.P50) } else { '' }
        '{0,-12} | {1,8} {2,8} {3,8} | {4,8} {5,8} {6,8} | {7,9}' -f $k,
            $(if ($pa) { '{0:N2}' -f $pa.P50 } else { '-' }), $(if ($pa) { '{0:N2}' -f $pa.P95 } else { '-' }), $(if ($pa) { $pa.Frames } else { '-' }),
            $(if ($pb) { '{0:N2}' -f $pb.P50 } else { '-' }), $(if ($pb) { '{0:N2}' -f $pb.P95 } else { '-' }), $(if ($pb) { $pb.Frames } else { '-' }),
            $ratio
    }
    ''
    'Level switches (ms)'
    $lv = @($a.Switches.Level) + @($b.Switches.Level) | Sort-Object -Unique
    foreach ($l in $lv) {
        $sa = ($a.Switches | Where-Object Level -eq $l | ForEach-Object Ms) -join ', '
        $sb = ($b.Switches | Where-Object Level -eq $l | ForEach-Object Ms) -join ', '
        '  -> L{0,-2}  A: {1,-24} B: {2}' -f $l, $sa, $sb
    }
    ''
    "Stroke wall ms   A: $(Stat ($a.Strokes | ForEach-Object WallMs))"
    "                 B: $(Stat ($b.Strokes | ForEach-Object WallMs))"
    "Pen-up ms        A: $(Stat ($a.Penups  | ForEach-Object Ms))"
    "                 B: $(Stat ($b.Penups  | ForEach-Object Ms))"
    return
}

# ---- run a build -------------------------------------------------------------

$exe = $exes[$Mode]
if (-not (Test-Path $exe)) { throw "Build not found: $exe (build $Mode first)" }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$safe  = ($Label -replace '[^\w.-]+', '_').Trim('_')
$name  = if ($safe) { "$stamp-$Mode-$safe" } else { "$stamp-$Mode" }
$dir   = Join-Path $runsDir $name
New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir 'log.txt'
$csv = Join-Path $dir 'frames.csv'

$env:CHISEL_PERF       = '1'
$env:CHISEL_PERF_CSV   = $csv
$env:CHISEL_DIRTY_HIST = '1'          # turns on the app's own [stage]/[penup] stroke timers
$env:CHISEL_PERF_VSYNC = $(if ($NoVsync) { '0' } else { '1' })

$appArgs = if ($Open) { "`"$((Resolve-Path $Open).Path)`"" } else { '' }
Write-Host "Running $Mode build -> $dir"
Write-Host 'Sculpt away; close the Chisel window to finish the run.'
$cmdline = "/c `"`"$exe`" $appArgs > `"$log`" 2>&1`""
$p = Start-Process cmd.exe -ArgumentList $cmdline -WorkingDirectory (Split-Path $exe) -WindowStyle Hidden -PassThru
$p.WaitForExit()
foreach ($v in 'CHISEL_PERF', 'CHISEL_PERF_CSV', 'CHISEL_DIRTY_HIST', 'CHISEL_PERF_VSYNC') { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }

$r = Read-Run $dir
$summary = Format-Run $r
$summary | Set-Content -Encoding utf8 (Join-Path $dir 'summary.txt')
''
$summary

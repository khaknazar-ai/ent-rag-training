param(
    [Parameter(Mandatory = $true)][string]$Path,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$target = (Resolve-Path -LiteralPath $Path).Path
$ownerSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$allowedSids = @(
    $ownerSid.Value,
    'S-1-5-18',     # LocalSystem
    'S-1-5-32-544'  # Built-in Administrators
)

if (-not $CheckOnly) {
    $acl = [System.IO.File]::GetAccessControl($target)
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($rule in @($acl.Access)) {
        $acl.RemoveAccessRuleSpecific($rule) | Out-Null
    }
    foreach ($sidText in $allowedSids) {
        $sid = [System.Security.Principal.SecurityIdentifier]::new($sidText)
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid,
            [System.Security.AccessControl.FileSystemRights]::FullControl,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
    }
    [System.IO.File]::SetAccessControl($target, $acl)
}

$current = [System.IO.File]::GetAccessControl($target)
$unexpected = @($current.Access | Where-Object {
    if ($_.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
        return $false
    }
    $sid = $_.IdentityReference.Translate(
        [System.Security.Principal.SecurityIdentifier]
    ).Value
    return $sid -notin $allowedSids
})
if (-not $current.AreAccessRulesProtected -or $unexpected.Count -gt 0) {
    Write-Error 'The local demo token file still has broad or inherited access.'
    exit 1
}

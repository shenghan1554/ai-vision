; AI Vision installer (Inno Setup 6)
; Builds a single setup EXE from the PyInstaller onedir output in dist\AIVision.

#define AppName "AI Vision"
#define AppVersion "2.6.0"
#define AppPublisher "AI Vision"
#define AppExeName "AIVision.exe"

[Setup]
AppId={{9B4C7C21-3E7A-4E2B-9C11-A1V1S10N2026}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\release
OutputBaseFilename=AI Vision-v2.6-Setup
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0
DisableDirPage=auto
DisableReadyPage=no
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
; The whole portable folder is installed; _internal carries all runtime DLLs.
Source: "..\dist\AIVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\使用说明"; Filename: "{app}\使用说明.md"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "运行 {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\runs"
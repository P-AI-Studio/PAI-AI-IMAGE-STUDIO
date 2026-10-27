#define AppName "P.AI AI IMAGE STUDIO"
#ifndef AppVersion
#define AppVersion "1.0.0-beta.2"
#endif
#define AppVerName "P.AI AI IMAGE STUDIO Version 1 Beta"
#define AppExeName "PAI-AI-Image-Studio.exe"
#define SourceDir AddBackslash(SourcePath) + "dist\PAI-AI-Image-Studio"

[Setup]
AppId={{D8D6F01D-DA6A-44F6-9D17-0A31AB0E76CA}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppVerName}
LicenseFile={#SourcePath}\LICENSE
AppPublisher=P.AI
DefaultDirName={localappdata}\Programs\P.AI\AI Image Studio
DefaultGroupName=P.AI AI IMAGE STUDIO
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64
DisableProgramGroupPage=yes
OutputDir=dist-installer
OutputBaseFilename=PAI-AI-Image-Studio-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
SetupIconFile={#SourcePath}\assets\pai_icon.ico
SetupLogging=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Dirs]
; Preserve user models, settings, cache, and outputs during uninstall.
Name: "{app}\data"; Flags: uninsneveruninstall

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\P.AI AI IMAGE STUDIO"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\P.AI AI IMAGE STUDIO"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch P.AI AI IMAGE STUDIO · Version 1 Beta"; Flags: postinstall nowait skipifsilent

; User data, models, caches, and outputs live under {app}\data and are preserved by uninstall.
; Remove that folder manually only when you intentionally want to delete user data.

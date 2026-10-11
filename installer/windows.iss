#define MyAppName "JARVIS-MARK-I"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Shashank HU"
#define MyAppExeName "JARVIS.exe"

[Setup]
AppId={{F6D2DCC4-3E85-4A5A-9DF0-41E7DB6F8A31}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\JARVIS-MARK-I
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=JARVIS-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\JARVIS.exe
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\JARVIS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\JARVIS-MARK-I"; Filename: "{app}\JARVIS.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\JARVIS-MARK-I"; Filename: "{app}\JARVIS.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\JARVIS.exe"; Description: "Launch JARVIS-MARK-I"; Flags: postinstall nowait skipifsilent

; NorthAI Installer 
; This installer downloads dependencies from official sources during
; installation, keeping the installer small and always up-to-date.
;
; Folder layout required:
;   Phase1-localised-gpt\
;       northai.iss        <-- Run InnoSetup from HERE
;       *.py
;       retrieval\*
;       specialists\*
;       models\
;           bge-small-en-v1.5\...
;
; Build output:
;   Installers\NorthAI-Setup.exe (~50-100MB)
;
; Downloads during installation:
;   - Python 3.12.7 (~30MB)
;   - Ollama (official Windows installer)
;   - Ollama models (via `ollama pull`)
; Bundled:
;   - Tesseract OCR installer (offline install)

[Setup]
AppName=NorthAI
AppVersion=1.0.0
AppPublisher=North AI
DefaultDirName={autopf}\NorthAI
DefaultGroupName=NorthAI
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
OutputBaseFilename=NorthAI-Setup
OutputDir=Installers
WizardStyle=modern
SetupLogging=yes
PrivilegesRequired=admin
ChangesEnvironment=yes
UninstallDisplayIcon={app}\launch_northai.py
SetupIconFile=compiler:SetupClassicIcon.ico
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Optional offline Python installer
Source: "Installers\payload\python-3.12.7-amd64.exe"; \
    DestDir: "{app}\installers"; Flags: ignoreversion skipifsourcedoesntexist

; Python source files
Source: "*.py"; DestDir: "{app}"; \
    Flags: ignoreversion; Excludes: "*.pyc __pycache__\* *.spec"

; Project packages
Source: "retrieval\*"; DestDir: "{app}\retrieval"; \
    Flags: recursesubdirs createallsubdirs ignoreversion
Source: "specialists\*"; DestDir: "{app}\specialists"; \
    Flags: recursesubdirs createallsubdirs ignoreversion

; Streamlit config
Source: ".streamlit\*"; DestDir: "{app}\.streamlit"; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist

; Requirements
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; BGE embedding model (bundled)
Source: "models\bge-small-en-v1.5\*"; DestDir: "{app}\models\bge-small-en-v1.5"; \
    Flags: recursesubdirs createallsubdirs ignoreversion

; Bundled Tesseract installer
Source: "Installers\payload\tesseract-ocr-w64-setup-5.5.0.20241111.exe"; \
    DestDir: "{app}\installers"; Flags: ignoreversion

[Dirs]
; App-local data
Name: "{code:RealLocalAppData}\NorthAI"
Name: "{code:RealLocalAppData}\NorthAI\web_rag_db"
Name: "{code:RealLocalAppData}\NorthAI\logs"

; NOTE:
; We no longer create {app}\ollama\models under Program Files.
; Ollama’s own installer will use:
;   - Binaries:  %LOCALAPPDATA%\Programs\Ollama
;   - Models:    %HOMEPATH%\.ollama\models
; which are user-writable and supported by the vendor.

[Icons]
Name: "{group}\NorthAI"; \
    Filename: "{app}\.venv\Scripts\python.exe"; \
    Parameters: """{app}\launch_northai.py"""; \
    WorkingDir: "{app}"
Name: "{autodesktop}\NorthAI"; \
    Filename: "{app}\.venv\Scripts\python.exe"; \
    Parameters: """{app}\launch_northai.py"""; \
    WorkingDir: "{app}"
Name: "{group}\Uninstall NorthAI"; Filename: "{uninstallexe}"

[Registry]
; IMPORTANT:
; We used to set OLLAMA_MODELS to {app}\ollama\models under Program Files,
; which caused permission errors like:
;   mkdir C:\Program Files\NorthAI\ollama\models\manifests: Access is denied
; This block has been REMOVED so Ollama uses its default model path under
; the user profile (C:\Users\<user>\.ollama\models on Windows).

; (no registry keys needed for Ollama now)

; ---------------------------------------------------------------------------
; INSTALLATION STEPS
; ---------------------------------------------------------------------------

[Run]
; 1) Use local Python installer if present
Filename: "{app}\installers\python-3.12.7-amd64.exe"; \
    Parameters: "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=0"; \
    StatusMsg: "Installing Python from offline package..."; \
    Flags: waituntilterminated; \
    Check: (not PythonExists) and FileExists(ExpandConstant('{app}\installers\python-3.12.7-amd64.exe'))

; 1b) Otherwise download Python (fallback only)
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe' -OutFile '{app}\python-installer.exe' -UseBasicParsing"""; \
    StatusMsg: "Downloading Python 3.12.7 (30MB)..."; \
    Flags: runhidden waituntilterminated; \
    Check: (not PythonExists) and (not FileExists(ExpandConstant('{app}\installers\python-3.12.7-amd64.exe')))

Filename: "{app}\python-installer.exe"; \
    Parameters: "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=0"; \
    StatusMsg: "Installing Python..."; \
    Flags: waituntilterminated; \
    Check: (not PythonExists) and (not FileExists(ExpandConstant('{app}\installers\python-3.12.7-amd64.exe')))

; 2) Create virtual environment (only if Python exists and version >= 3.12)
Filename: "{cmd}"; \
    Parameters: "/C python -m venv ""{app}\.venv"""; \
    StatusMsg: "Creating virtual environment..."; \
    Flags: runhidden waituntilterminated; \
    Check: PythonExists and PythonCorrectVersion

; 3) Upgrade pip & install Python packages
Filename: "{app}\.venv\Scripts\python.exe"; \
    Parameters: "-m pip install --upgrade pip setuptools wheel"; \
    StatusMsg: "Upgrading pip..."; \
    Flags: runhidden waituntilterminated; \
    Check: FileExists(ExpandConstant('{app}\.venv\Scripts\python.exe'))

; Install Python dependencies
Filename: "{app}\.venv\Scripts\python.exe"; \
    Parameters: "-m pip install -r ""{app}\requirements.txt"""; \
    StatusMsg: "Installing Python packages (this may take a few minutes)..."; \
    Flags: runhidden waituntilterminated

; 3b) Install Python Ollama SDK (force correct version)
Filename: "{app}\.venv\Scripts\python.exe"; \
    Parameters: "-m pip install --force-reinstall --no-cache-dir ollama"; \
    StatusMsg: "Installing Ollama Python SDK..."; \
    Flags: runhidden waituntilterminated

; 3c) Remove conflicting system-level Ollama Python module (prevents wrong import)
Filename: "{cmd}"; \
    Parameters: "/C if exist ""{commonpf64}\Ollama\ollama.py"" ren ""{commonpf64}\Ollama\ollama.py"" ""ollama_cli_unused.py"""; \
    StatusMsg: "Removing conflicting Ollama module..."; \
    Flags: runhidden waituntilterminated

; 4) Install Tesseract OCR (bundled — no download)
Filename: "{app}\installers\tesseract-ocr-w64-setup-5.5.0.20241111.exe"; \
    Parameters: "/VERYSILENT /NORESTART /DIR=""{commonpf}\Tesseract-OCR"""; \
    StatusMsg: "Installing Tesseract OCR..."; \
    Flags: waituntilterminated; \
    Check: not TesseractExists

; 5) Use local Ollama installer if present
; NOTE:
; OllamaSetup.exe installs per-user under:
;   %LOCALAPPDATA%\Programs\Ollama
; and adds `ollama` to that user’s PATH, and models to %HOMEPATH%\.ollama\models.
Filename: "{app}\installers\OllamaSetup.exe"; \
    Parameters: "/VERYSILENT /NORESTART"; \
    StatusMsg: "Installing Ollama from offline package..."; \
    Flags: waituntilterminated; \
    Check: (not OllamaExists) and FileExists(ExpandConstant('{app}\installers\OllamaSetup.exe'))

; 5b) Otherwise download Ollama (fallback only)
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile '{tmp}\ollama-installer.exe' -UseBasicParsing"""; \
    StatusMsg: "Downloading Ollama (500MB)..."; \
    Flags: runhidden waituntilterminated; \
    Check: (not OllamaExists) and (not FileExists(ExpandConstant('{app}\installers\OllamaSetup.exe')))

; 5c) Install Ollama from temp directory
Filename: "{tmp}\ollama-installer.exe"; \
    Parameters: "/VERYSILENT /NORESTART"; \
    StatusMsg: "Installing Ollama..."; \
    Flags: waituntilterminated; \
    Check: (not OllamaExists) and (not FileExists(ExpandConstant('{app}\installers\OllamaSetup.exe')))

; 6) Clean up Ollama auto-start + stray processes
;    - Remove Startup shortcut, if any
;    - Kill running ollama.exe so model pull won't hit a busy port
Filename: "{cmd}"; \
    Parameters: "/C del ""{userstartup}\Ollama.lnk"" 2>nul"; \
    StatusMsg: "Cleaning Ollama startup shortcut (user)..."; \
    Flags: runhidden

Filename: "{cmd}"; \
    Parameters: "/C del ""{commonstartup}\Ollama.lnk"" 2>nul"; \
    StatusMsg: "Cleaning Ollama startup shortcut (common)..."; \
    Flags: runhidden

Filename: "{cmd}"; \
    Parameters: "/C taskkill /IM ollama.exe /F 2>nul"; \
    StatusMsg: "Stopping running Ollama processes..."; \
    Flags: runhidden

; 6c) Start Ollama server manually (professional reliability fix)
Filename: "{cmd}"; \
Parameters: "/C start """" ""{localappdata}\Programs\Ollama\ollama.exe"" serve"; \
StatusMsg: "Starting Ollama server..."; \
Flags: runhidden


; 6d) Wait for Ollama API to come online (PowerShell version - reliable)
Filename: "powershell.exe"; \
Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""For ($i=0; $i -lt 90; $i++) {{ try {{ iwr http://127.0.0.1:11434 -useBasicParsing | Out-Null; exit }} catch {{ ""Waiting for Ollama to become available...""; }} Start-Sleep 2 }} Exit 1"""; \
StatusMsg: "Waiting for Ollama to become available..."; \
Flags: runhidden waituntilterminated


; 7) Pull required Ollama models (using default models path under the user profile)
Filename: "{cmd}"; \
    Parameters: "/C ollama pull llama3:8b"; \
    StatusMsg: "Downloading AI models (this may take 5–10 minutes)..."; \
    Flags: runhidden waituntilterminated; \
    Check: OllamaExists and OllamaWorking

; 8) Verify BGE model exists
Filename: "{cmd}"; \
    Parameters: "/C if exist ""{app}\models\bge-small-en-v1.5\config.json"" (exit /b 0) else (exit /b 1)"; \
    StatusMsg: "Verifying embedding model..."; \
    Flags: runhidden waituntilterminated

; 9) Launch app after install
Filename: "{app}\.venv\Scripts\python.exe"; \
    Parameters: """{app}\launch_northai.py"""; \
    WorkingDir: "{app}"; \
    Description: "Launch NorthAI"; \
    Flags: nowait postinstall skipifsilent

[Code]

function RealLocalAppData(Param: String): String;
begin
  Result := GetEnv('LOCALAPPDATA');
end;

function PythonExists: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(
    ExpandConstant('{cmd}'),
    '/C python --version',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);

  if Result then
    Log('Python is already installed')
  else
    Log('Python not found, will download and install');
end;

function TesseractExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{commonpf}\Tesseract-OCR\tesseract.exe')) or
            FileExists(ExpandConstant('{commonpf32}\Tesseract-OCR\tesseract.exe'));

  if Result then
    Log('Tesseract is already installed')
  else
    Log('Tesseract not found, will install bundled version');
end;

function OllamaExists: Boolean;
var
  ResultCode: Integer;
begin
  // Checks whether 'ollama' CLI is available on PATH for the installing user.
  Result := Exec(
    ExpandConstant('{cmd}'),
    '/C ollama --version',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);

  if Result then
    Log('Ollama is already installed')
  else
    Log('Ollama not found, will download and install');
end;

function PythonCorrectVersion: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(
    ExpandConstant('{cmd}'),
    '/C python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)"',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);
end;

function OllamaWorking: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(
    ExpandConstant('{cmd}'),
    '/C ollama list',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);
end;

function TesseractWorking: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec(
    ExpandConstant('{cmd}'),
    '/C tesseract --version',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    Log('Installation completed successfully');
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  Log('NorthAI Setup started');
  Log('Installation will download required dependencies from official sources');
end;

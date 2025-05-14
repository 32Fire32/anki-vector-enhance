@echo off
REM Ottiene il percorso della cartella dove si trova questo .bat
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo -------------------------------
echo Installing Vector Supervisor Service...
echo -------------------------------

REM Percorso completo a Python 3.9
set PYTHON_EXE="C:\Users\nicol\AppData\Local\Programs\Python\Python39\python.exe"

REM Installazione del servizio
%PYTHON_EXE% install_service.py install
if errorlevel 1 (
    echo Errore durante l'installazione del servizio.
    pause
    exit /b 1
)

REM Configura l'avvio automatico del servizio
sc config VectorSupervisorService start= auto

REM Avvio del servizio
%PYTHON_EXE% install_service.py start
if errorlevel 1 (
    echo Errore durante l'avvio del servizio.
    pause
    exit /b 1
)

echo -------------------------------
echo Il servizio è stato installato e avviato con successo.
echo -------------------------------
pause

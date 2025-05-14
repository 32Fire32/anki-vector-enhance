@echo off
echo -------------------------------
echo Uninstalling Vector Supervisor Service...
echo -------------------------------
cd /d %~dp0

REM Esegui il comando di disinstallazione
python install_service.py remove

if %ERRORLEVEL% neq 0 (
    echo Errore durante la disinstallazione del servizio.
) else (
    echo -------------------------------
    echo Il servizio è stato disinstallato con successo.
    echo -------------------------------
)

pause

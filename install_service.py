import win32serviceutil
import win32service
import win32event
import servicemanager
import os
import sys

class VectorService(win32serviceutil.ServiceFramework):
    _svc_name_ = "VectorSupervisorService"
    _svc_display_name_ = "Vector Supervisor for Anki Robot"
    _svc_description_ = "Supervises Anki Vector events and logs them to a SQL Server database."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.running = False
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogInfoMsg("Vector Supervisor Service Started")
        exec(open("service/supervisor.py").read())  # Esegue lo script

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(VectorService)

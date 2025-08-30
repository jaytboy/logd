# Log'd (Lite)
## Time Tracker App (PyQt6)

A **Windows desktop time-tracking application** built with **Python 3** and **PyQt6**, using **SQLite** for local data storage.  
Designed to track time spent on tasks and associate tasks with projects. Includes reporting, CSV export, and a system tray integration.

---

## Features

### Tasks
- Add tasks (unique **per project**, case-insensitive).  
- Soft-delete tasks (archived) without affecting historical sessions.  
- Start, pause, switch, and resume tasks.  
- 1-second timer updates elapsed time live.  
- End Day button resets active tasks and timer.  
- Tasks sorted by **total time spent in the last 7 days**.

### Projects
- Add projects with detailed information:
  - Name, Number, Location (City + State), Client
  - Type: New / Remodel
  - Scope area (sqft)
  - Disciplines: HVAC and/or Plumbing
- Only delete projects that have **no tasks**.  
- Tasks are associated with a single project.  

### Reports
- Filter by start and end date.  
- Display session details:
  - Session ID, Task, Start Time, End Time, Elapsed Time.  
- Export CSV of current report.  

### System Tray
- Tray icon with context menu to start/pause tasks or quit.  
- Double-click tray icon to show/raise main window.  

### UI & UX
- Timer label shows **HH:MM:SS** format.  
- Drop-down lists auto-resize to fit content.  
- Always-on-top toggle.  
- Modal dialogs for warnings, confirmations, and errors.  

---

## Installation

### Requirements
- Windows 10/11  
- Python 3.9+  
- Packages:
  ```bash
  pip install PyQt6

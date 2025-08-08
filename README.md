<h3 style="color: orange; font-weight: bold;">
🚧 This project is currently under active development
</h3>

1️⃣ Download the Project
If it’s already on GitHub:

Go to your GitHub repository link.

Click Code → Download ZIP.

Extract the ZIP file somewhere, e.g.

makefile
Copy code
C:\Users\darsh\Desktop\BrainStroke
If it’s not on GitHub yet, just copy your current folder and give it to someone.

2️⃣ Install Python
You (and anyone running it) should have Python 3.10+ installed.

Recommended: Install Python 3.11.x from
https://www.python.org/downloads/
✅ During installation, check “Add Python to PATH”.

3️⃣ Open Command Prompt in Project Folder
Press Shift + Right-click inside the project folder → Open PowerShell window here.

Or, in Command Prompt:

powershell
Copy code
cd "C:\Users\darsh\Desktop\BrainStroke"
4️⃣ Create a Virtual Environment (Recommended)
powershell
Copy code
python -m venv venv
Activate it:

powershell
Copy code
venv\Scripts\activate
5️⃣ Install Requirements
If you have a requirements.txt file:

powershell
Copy code
pip install -r requirements.txt
⚠ If there are any install errors, install missing packages manually, for example:

powershell
Copy code
pip install flask torch torchvision
6️⃣ Run the Project
powershell
Copy code
python app.py
If everything is fine, you’ll see:

csharp
Copy code
 * Running on http://127.0.0.1:5000
7️⃣ Open in Browser
Go to:

cpp
Copy code
http://127.0.0.1:5000
The Brain Stroke Detection web app will open.

8️⃣ Stop the App
Press CTRL + C in the terminal.

💡 Extra Tip — Double Click to Run
If you want to run it without typing commands:

Create a new text file run.bat in the project folder.

Add:

bat
Copy code
@echo off
call venv\Scripts\activate
python app.py
pause
Save and double-click run.bat anytime.

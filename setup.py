import cx_Freeze 
import sys,os
import matplotlib
base = None
os.environ['TCL_LIBRARY'] = 'C:/Users/002/Anaconda3/pkgs/tk-8.6.8-hfa6e2cd_0/Library/lib/tcl8.6'
os.environ['TK_LIBRARY'] = 'C:/Users/002/Anaconda3/pkgs/tk-8.6.8-hfa6e2cd_0/Library/lib/tk8.6'
if sys.platform=='win32':
	base='Win32GUI'
executables=[cx_Freeze.Executable("project.py",base=base,icon="icon.ico")]
cx_Freeze.setup(
	name="ProjectTiger",
	options={"build_exe":{
	"packages":["tkinter","Tkinter","sklearn","keras","tensorflow","PIL","glob","os","pygame","matplotlib","sys","numpy","shutil"],
	"include_files":["C:/Users/002/Anaconda3/pkgs/tk-8.6.8-hfa6e2cd_0/Library/bin/tcl86t.dll","C:/Users/002/Anaconda3/pkgs/tk-8.6.8-hfa6e2cd_0/Library/bin/tk86t.dll","icon.ico","logo.gif","keras_model.h5"]
	}},
	version="1.0",
	description="projectS8",
	executables=executables
	)


 
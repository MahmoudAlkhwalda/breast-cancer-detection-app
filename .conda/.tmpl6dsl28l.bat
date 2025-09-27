@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\USER\anaconda3\condabin\conda.bat" activate "c:\Users\USER\OneDrive\سطح المكتب\breast_cancer_app\.conda"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@"c:\Users\USER\OneDrive\سطح المكتب\breast_cancer_app\.conda\python.exe" -Wi -m compileall -q -l -i C:\Users\USER\AppData\Local\Temp\tmp5ip5c88h -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL

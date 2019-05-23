set root=%HOMEPATH%\Anaconda3\

call %root%\Scripts\activate.bat %root%

echo %cd%
call conda activate geo_env
jupyter-notebook


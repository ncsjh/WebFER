from cx_Freeze import setup, Executable

setup(name= 'Emotion Recognition',
       version='1.0',
       executables = [Executable('__flask.py')])
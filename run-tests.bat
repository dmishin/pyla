@echo off
for %%i in (test_*.py) do (
    echo Running %%i
    py25 %%i
    )


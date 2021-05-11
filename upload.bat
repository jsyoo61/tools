D:
cd programming\tools
START /WAIT python setup.py sdist bdist_wheel
START /WAIT python -m twine upload dist/*
START /WAIT pip install --upgrade tools-jsyoo61
DEL dist\* /Q
PAUSE

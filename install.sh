#clear terminal
clear

#execute prebuild script
python scripts/prebuild.py

# Check for command line arguments
if [ "$1" == "--editable" ]; then
    #remove already present builds (if any)
    rm -rf build
    rm -rf *.egg-info

    #run the installation
    pip install -e .
else
    #run the installation
    pip install .
fi
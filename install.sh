#clear terminal
clear

#remove alredy builds
rm -rf build
rm -rf *.egg-info

python scripts/prebuild.py

#run the installation
pip install -e .
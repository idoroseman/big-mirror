# installation
note: tensorflow runs on python 3.12 or less
for me 3.9 works best 

# clone repo
git clone https://github.com/idoroseman/big-mirror
cd big-mirror

## create venv
python3 -m venv venv

## activate venv
source venv/bin/activate

## install libraries
pip install deepface
pip install elevenlabs
pip install pygame

on intel cpu I had to pip install "numpy<2.0" 

## fill the database directory
database/
+ {name}/
| + name.txt  (optional)
| + {pitures...}
+ {name}/
...

## prepare audio
python3 prep_audio.py



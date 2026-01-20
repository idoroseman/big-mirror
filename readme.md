# installation

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

## fill the database directory
database/
+ {name}/
| + name.txt  (optional)
| + {pitures...}
+ {name}/
...

## prepare audio
python3 prep_audio.py



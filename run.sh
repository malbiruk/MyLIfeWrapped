#!/usr/bin/env bash

source /home/klim/.virtualenvs/MyLifeWrapped/bin/activate
cd /home/klim/Documents/work/scripts/MyLifeWrapped/

if [ -n "$1" ]; then
    python -m calendar_.obtain_data "$1"
else
    python -m calendar_.obtain_data
fi

python get_stats.py
python upload_imgs.py
python update_slides.py

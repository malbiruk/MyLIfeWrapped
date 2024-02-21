#!/usr/bin/env bash

source /home/klim/.virtualenvs/MyLifeWrapped/bin/activate
cd /home/klim/Documents/work/scripts/MyLifeWrapped/calendar_

if [ -n "$1" ]; then
    python obtain_data.py "$1"
else
    python obtain_data.py
fi

cd ..
python get_stats.py
python upload_imgs.py
python update_slides.py

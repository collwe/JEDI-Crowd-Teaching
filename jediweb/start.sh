#!/bin/bash

PYTHONPATH=/home/stethox/JEDI_KDD18/jediweb
source /home/stethox/JEDI_KDD18/.venv/bin/activate
cd /home/stethox/JEDI_KDD18/jediweb
/home/stethox/JEDI_KDD18/.venv/bin/gunicorn --workers 6 --bind 198.11.228.162:9000 jediweb.wsgi

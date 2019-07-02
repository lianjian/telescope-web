#!/bin/bash
echo "Activating venv"
. venv/bin/activate

echo "export FLASK_APP=telescope_flask"
export FLASK_APP=telescope_flask

echo "export FLASK_ENV=development"
export FLASK_ENV=development

flask run

#!/bin/bash
mkdir public_html

# Executing SPHINX docs

cd docs
sphinx-build -b html ./ ../public_html || exit 1;
cd ..

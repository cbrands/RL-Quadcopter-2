#!/usr/bin/env bash

echo "-- Installing packages"
apt-get update
apt-get install nano
pip install -r requirements.txt

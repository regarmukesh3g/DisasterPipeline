#!/usr/bin/env python
"""
A Flask Webapp for Disaster Response.
"""

from flask import Flask

app = Flask(__name__)
from webapp import routes


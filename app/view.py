from flask import render_template

def index():
  return render_template('index.html')

def app():
  return render_template('app.html')

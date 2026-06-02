from flask import Flask
from app import view

app = Flask(__name__)

app.add_url_rule(rule= '/', endpoint = 'home', view_func = view.index)
app.add_url_rule(rule= '/app', endpoint = 'app', view_func = view.app)

if __name__ == '__main__':
  app.run(debug=True)

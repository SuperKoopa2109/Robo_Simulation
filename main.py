import argparse
#import flask
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/simulation")
def simulation():
    return render_template("robo_sim_widget.html")

def init_env():
    parser = argparse.ArgumentParser()

    parser.add_argument('--webserver', default='True', help='Argument for whether this program should run a webserver or not. Alternatively this can be run without a UI.')
    parser.add_argument('--port', type=int, default='8000', help='provide port for webserver')

    FLAGS = parser.parse_args()

    print(f'webserver{FLAGS.webserver}')    
    print(f'port: {FLAGS.port}')

if __name__ == '__main__':
    init_env()
    app.run(debug=True)







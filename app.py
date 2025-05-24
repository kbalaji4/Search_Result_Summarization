from flask import Flask, request, render_template
from demo_5 import process, process_follow_up_input, handle_reset

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return process()

@app.route('/follow-up', methods=['POST'])
def follow_up():
    return process_follow_up_input()

@app.route('/reset', methods=['POST'])
def reset():
    return handle_reset()  

if __name__ == '__main__':
    app.run()
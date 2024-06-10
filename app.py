from flask import Flask, request, render_template
from mlx_lm import load, generate

app = Flask(__name__)

# Load the model and tokenizer once at the start
model, tokenizer = load("mlx-community/OpenHermes-2.5-Mistral-7B-4bit-mlx")
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        response = generate(model, tokenizer, prompt)
        return render_template('index.html', prompt=prompt, response=response)
    return render_template('index.html', prompt='', response='')

if __name__ == '__main__':
    app.run(debug=True)
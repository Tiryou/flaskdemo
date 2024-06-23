from flask import Flask,request,render_template
import demo_with_gradio

app = Flask(__name__)

global embed_embeddings
embed_embeddings = []
global embed_file_names
embed_file_names = []
global user_names
user_names = []
global names
names = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        embed_file = request.files['embed_file']
        global embed_file_names
        embed_file_names.append(embed_file.filename)
        global embed_embeddings
        embed_embeddings.append(demo_with_gradio.get_embeddings(embed_file))
        global user_names
        user_names.append(request.form['username'])
        global names
        names = zip(user_names, embed_file_names)
    return render_template('index1.html', names=names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        eval_file = request.files['eval_file']
        eval_file_name=eval_file.filename
        eval_embedding = demo_with_gradio.get_embeddings(eval_file)
        threshold = 0.54567164
        score=0.0
        scores=[]
        for embed_embedding in embed_embeddings:
            score = demo_with_gradio.detect_same_speaker_with_embeddings(embed_embedding, eval_embedding)
            scores.append(score)
        max_score = max(scores)
        max_speaker = user_names[scores.index(max_score)]
        if max_score > threshold:
            speaker = max_speaker
        else:
            speaker = "Unknown"
        global names
        names = list(zip(user_names, embed_file_names))
        return render_template('index1.html', score=max_score, speaker=speaker, max_speaker=max_speaker, eval_file_name=eval_file_name, names=names)

if __name__ == '__main__':
    app.run()
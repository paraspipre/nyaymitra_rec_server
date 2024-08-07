# !pip install flask_cors pymongo pyngrok
# !ngrok config add-authtoken 2b3XUfLVlKvSWUipBL7Ahw6gx61_453MywRgQZG3DAFHAeu7B
from flask import Flask , request , jsonify
from flask_cors import CORS, cross_origin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
# from pyngrok import ngrok
import spacy

#create instance of the SpaCy NLP model
nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)
CORS(app)


@app.route("/check",methods=["GET"])
def check():
   return {"message":"success"}

client = MongoClient('mongodb+srv://paras:paras@cluster0.uaueb.mongodb.net')
db = client['test']
collection = db['users']

@app.route('/recommadlawyer', methods=['POST'])
@cross_origin()
def recommand():
   data = request.json
   text = data.get('text')
   def preprocess_text(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in nlp.Defaults.stop_words]
    return ' '.join(lemmatized_tokens)

   processed_fir = preprocess_text(text)

   lawyers = collection.find({"role": 1})
   data = []
   for document in lawyers:
    data.append({
        'id': str(document['_id']),
        'name': document['name'],
        'tags': document['tags'],
    })
    # Return the data as JSON

   documents = [processed_fir] + list([d['tags'] for d in data])

   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform(documents)

   similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

   print(sorted(similarities, reverse=True))
   sorted_indices = np.argsort(similarities)[::-1]
   top_n_indices = sorted_indices[:10]
   print(top_n_indices)
   recommended_lawyers = []
   for index in top_n_indices:
    recommended_lawyers.append(data[index])
   print(recommended_lawyers)
   return jsonify(recommended_lawyers)

if __name__ == '__main__':
   #  ngrok_tunnel = ngrok.connect(5000)
   #  print('Public URL:', ngrok_tunnel.public_url)
    app.run(port=5000)
from flask import * 
from tkinter import messagebox
from pyvi import ViTokenizer
import gensim
from joblib import load
from tensorflow import keras
 
# Tải lại mô hình
ifidf = load('E:/CodeDatabase/Web/Natual-Language-Processing/Text-Classifier/model/TF-IDF_Char.joblib')
svd = load('E:/CodeDatabase/Web/Natual-Language-Processing/Text-Classifier/model/SVD_Char.joblib')
encoder = load('E:/CodeDatabase/Web/Natual-Language-Processing/Text-Classifier/model/Encoder.joblib')

#Load model RNCC
model = keras.models.load_model('E:/CodeDatabase/Web/Natual-Language-Processing/Text-Classifier/model/RCNN.h5')

#Tiền xử lý dữ liệu
def preprocessing_doc(doc):
      lines = gensim.utils.simple_preprocess(doc)
      lines = ' '.join(lines)
      lines = ViTokenizer.tokenize(lines)
      test_doc_tfidf = ifidf.transform([lines])
      test_doc_svd = svd.transform(test_doc_tfidf)
    
      return test_doc_svd

def check_genre(doc):

      vec = preprocessing_doc(doc)

      predictions = model.predict(vec)
      predictions = predictions.argmax(axis=-1)

      return encoder.classes_[predictions][0]


app = Flask(__name__)  


@app.route('/')  
def message():  
      return render_template('index.html')  
  
@app.route('/demo',methods = ['POST'])
def login():
      message=request.form['message']  
      print(message) 
      return render_template('demo.html',result = check_genre(message),message=message)  
   
if __name__ == '__main__':  
   app.run(debug = True) 
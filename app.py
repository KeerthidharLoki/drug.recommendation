
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle




app = Flask(__name__)

cosine_sim=pd.read_csv('E:\Projects\Drug\drug.recommendation\Files\cos_similarity.csv')
main=pd.read_csv('E:\Projects\Drug\drug.recommendation\Files\main.csv')
Reviews=pd.read_csv('E:\Projects\Drug\drug.recommendation\Files\Reviews.csv')

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the drug that matches the title
    idx=main.loc[main['DrugName']==title].index[0]
    #idx=692
    # Get the pairwsie similarity scores of all drugs with that drug
    sim_scores =list(enumerate(cosine_sim.loc[idx]))
    # Sort the drugs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar drugs
    sim_scores = sim_scores[1:10]
    # Get the drug indices
    Drug_indices = [i[0] for i in sim_scores]
     #Sorting again based on Review Rating and Count_of_Reviews
    Drug_indices=main.iloc[Drug_indices][main['Count_of_Reviews']>67] .sort_values(by='User_Rating',ascending=False).index
    # Return the top 10 most similar drugs
    return main['DrugName'].iloc[Drug_indices]

    


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/search_api",methods=['POST'])

def search_api():
    data=request.form['data']
    data=str(data)
    output=get_recommendations(data,cosine_sim)
    output=str(output)
    return output




if __name__ == '__main__':
    app.run(debug=True)
import json
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


def main(args):
    input_ing = []
    for i in args.ingredient:
        input_ing.append(i)
    input_string = ""
    input_final = []

    for x, y in enumerate(input_ing):
        input_string = input_string + y + " "
    input_final.append(input_string)
    cuisine_predictor(args.N,input_final)

def cuisine_predictor(N,input_final):
    data = readfile()
    cuisine, test_vector, Ingredients = ModelBuild(data,input_final)
    df , distance_closest = cosine_similarity_score(Ingredients , test_vector, data ,N)

    # create array for writing the output using dictionaries
    output_Values = df.apply(pd.Series.explode).to_dict(orient='records')
    output = {
        'cuisine': cuisine[0],
        'score': distance_closest,
        'closest': output_Values
    }
    print(json.dumps(output, indent=1))


def readfile():
    fname = "../cs5293sp23-project2/docs/yummly.json"
    data = pd.read_json(fname)
    return data

def ModelBuild(data, input_final):
    #take a list of ingredients string with their indexes
    ingredient_list = data['ingredients']
    #convert the list into string
    ing = []
    for l in ingredient_list:
        ingredient_string = [','.join(l).lower()]
        ing += ingredient_string
    #Assign string of ingredients to a dataframe with stringIngredients
    data['stringIngredients'] = ing
    #creating a TFDIF Vectorizer model
    vector1 = TfidfVectorizer()
    Ingredients = vector1.fit_transform(ing)
    #convert list of cuisine types into a string
    labels = data['cuisine']
    cusi = []
    for l in labels:
        label_string = [''.join(l).lower()]
        cusi += label_string
    # using label_values to assign labels to cuisine
    cuisine_values = LabelEncoder().fit_transform(data['cuisine'])
    # Use LogisticRegression Model to train the data
    clf = LogisticRegression(max_iter=100000)
    # Use LogisticRegression to fit cuisine values and Ingredients
    clf.fit(Ingredients, cuisine_values)

    vectors_transform = vector1.transform(input_final)
     #coverting vectors to an array
    test_vector = vectors_transform.toarray()
    #predicting and testing input vectorizations 
    test_preds = clf.predict(test_vector)
    #Testing using inverse transform for getting the values of the labels of the cuisines
    cuisine = LabelEncoder().fit(cusi)
    cuisine_list= cuisine.inverse_transform(test_preds).tolist()

    return cuisine_list, test_vector , Ingredients

def cosine_similarity_score(Ingredients , test_vector, data ,N):
    #finding the similarity score between each ingredient score
    similarityValues = cosine_similarity(test_vector[0:1],Ingredients)
    #giving the dataframe a similarity score and assigning the values
    data["score"] = similarityValues[0]
    #sort the similarity list and get the closest distance
    closestdistance = sorted(similarityValues[0])[-1]
    #Assign a datafram to the scorer of the closest distance
    data = data.sort_values(by=['score'], ascending=False)
    #getting the nearest neighbour for N
    data = data.head(N+1)
    #drop the closest row
    data = data.iloc[1:, :]
    closestneighbors = []
    cosinescore = []
    for index, row in data.iterrows():
        closestneighbors.append(row['id'])
        cosinescore.append(row['score'])
    df = pd.DataFrame({"id": closestneighbors, "score": cosinescore})
    return df , closestdistance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient",type=str,required=True,help="input ingredients",action="append")
    parser.add_argument("--N", type=int, required=True, help="nearest neighbors")
    args = parser.parse_args()
    main(args)

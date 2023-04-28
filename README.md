
#### Author: Chandra Likhitha Chopparapu

# Cuisine Predictor

The Project is implemented in Python and it can be run using this command.

pipenv run python project2.py --N 5 --ingredient paprika
                                    --ingredient banana 
                                    --ingredient "rice krispies" 
Tree structure for the files in Cuisine Predictor are :
<img width="392" alt="treesnippet" src="https://user-images.githubusercontent.com/51265009/235031190-9dba89d5-1c4e-42dd-8c4b-613690d9ab9f.PNG">


Different kinds of Packages and libraries used are:

argparse
pytest 

and some machine learning packages are

scikit-learn
numpy
pandas

Function and Features used are :

#### 1.readfile() 
 I have used this method to read the yummly dataset and then gave the path of the file where is it located and since it is a json file I have used read_json method from the pandas package to read the yummly.json dataset of cuisines names and respective ingredients list.

#### 2. ModelBuild()
I have used LogisticRegression to train and test Ingredients and Cuisine names. I first converted ingredients which are in the form of a list and then converted those to string and then put those strings in a dataframe. In the similar way I have converted Cuisine names into a List and then converted to a string and then put in a dataframe. I used TfidfVectorizer and Logistic Regression to fit the cuisine names and Ingredients and then tested those useing predict method and after that used LabelEncoder to get the Labels for the cuisines and then used inverseTransform to get the cuisine names.
This is the method where most of the scikit-learn methods are used.

#### 3. Cosine_similarity_ Score()
I have used this method to measure similarity between the scores of Cuisines and Ingredients and then get the approximate score for those to see how much they are similar to each other.The resulting value ranges from -1 to 1, with 1 indicating that the two vectors are identical, 0 indicating that the two vectors are orthogonal (i.e., have no similarity), and -1 indicating that the two vectors are diametrically opposed.
#### 4 . Cuisine_Predictor()
I have combined all the methods and called all the above mentioned into this one and this method will be used for giving out the output statements.

#### Bugs and Assumptions

Ingredient list errors: One common bug is the presence of errors or inconsistencies in the ingredient list of a recipe, such as misspellings or incomplete ingredients. This can lead to incorrect or incomplete feature extraction, and hence, incorrect cuisine predictions.

Ingredient representation: Another assumption is the representation of ingredients as binary features, i.e., either present or absent. However, this representation does not consider the quantity or proportion of each ingredient used in a recipe, which can significantly affect the flavor and texture of the dish, and hence, the cuisine to which it belongs.

Regional variations: Cuisine prediction assumes that the cooking methods, ingredients, and flavor profiles of a particular cuisine are consistent across regions. However, this assumption does not hold in reality, as many cuisines have significant regional variations due to differences in climate, culture, and local ingredients.

Multicultural dishes: Another assumption is that dishes belong to only one cuisine, whereas many dishes are a fusion of multiple cuisines, or have been adapted to suit the tastes of a different region or culture. This can make it difficult to assign a particular cuisine label to such dishes.

Limited training data: Finally, cuisine prediction relies heavily on the availability and quality of training data, which may not always be representative of the diversity and complexity of global cuisine. This can lead to biases and errors in the predictions, especially for less common or underrepresented cuisines.

#### Testfunctions

test_read()
In this test_cuisine.py file , I have created test_read() method which checks if all the functions are working properly in the project2.py file.
We have run the command
pipenv run python -m pytest and if the test passes then it means all the tests have been passed.

#### Sample output

{
  "cuisine": "America",
  "score": 0.91,
  "closest": [
    {
      "id": "10232",
      "score": 0.34
    },
    {
      "id": "10422",
      "score": 0.15
    },
    {
      "id": "45",
      "score": 0.13
    },
    {
      "id": "7372",
      "score": 0.04
    },
    {
      "id": "9898",
      "score": 0.02
    }
  ]

URL for gif Output is : https://github.com/Likhitha16/cs5293sp23-project2/blob/main/ezgif.com-video-to-gif%20(1).gif





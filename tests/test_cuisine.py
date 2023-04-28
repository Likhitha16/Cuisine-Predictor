import project2
import pytest
def test_read():
    data = project2.readfile()
    assert data is not None
    cuisine_values , vector_test_array , Ingredient_values = project2.ModelBuild(data,['banana'])
    assert cuisine_values is not None
    assert vector_test_array is not None
    assert Ingredient_values is not None
    a,b = project2.cosine_similarity_score(Ingredient_values,vector_test_array,data,7)
    assert a is not None
    assert b is not None


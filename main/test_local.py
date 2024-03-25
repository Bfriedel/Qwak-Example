from qwak.model.tools import run_local
from model import MovieRecommendation
import pandas as pd

if __name__ == '__main__':
    # Create a new instance of the model
    m = MovieRecommendation()
    
    # Create an input vector and convert it to JSON
    input_vector = pd.DataFrame([{"uid": "143362", "iid": "se7en+1995"}]).to_json()
    
    print(input_vector)
    
    # Run local inference using the model
    prediction = run_local(m, input_vector)
    print(prediction)

import os
import pandas as pd
import qwak
from surprise import Dataset, Reader, SVD, accuracy
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, InferenceOutput, ModelSchema
from surprise.model_selection import train_test_split as surprise_train_test_split

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

class MovieRecommendation(QwakModel):
    '''Movie Recommendation model.

    This class implements a movie recommendation model using Singular Value Decomposition (SVD)
    algorithm from the Surprise library.
    '''

    def __init__(self):
        '''Initialize the MovieRecommendation model.'''
        self.params = {
            "random_state": int(os.getenv("random_state", 0)),
            "n_factors": int(os.getenv("n_factors", 200)),
            "n_epochs": int(os.getenv("n_epochs", 20))
        }
        self.model = SVD(**self.params)
        qwak.log_param(self.params)

    def build(self):
        '''Build and train the movie recommendation model.'''
        #load data
        df = pd.read_csv(f"{RUNNING_FILE_ABSOLUTE_PATH}/ratings.csv", index_col=0)
        #dftop = df.head(100000)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userid', 'movieid', 'rating']], reader)

        #split to train and test
        trainset, testset = surprise_train_test_split(data, test_size=0.25, random_state=self.params["random_state"])

        #fit model
        self.model.fit(trainset)

        predictions = self.model.test(testset)

        #log metrics
        qwak.log_metric({"RMSE": accuracy.rmse(predictions), "MSE": accuracy.mse(predictions), 
                         "MAE": accuracy.mae(predictions), "FCP":accuracy.fcp(predictions)})


    def schema(self):
        '''Define inputs and outputs schema for the model.'''
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="uid", type=str),
                ExplicitFeature(name="iid", type=str),
            ],
            outputs=[InferenceOutput(name="Estimated_Rating", type=float)],
        )
        return model_schema
    
                
    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Predict the rating for a given user and movie.

        Args:
            df (pd.DataFrame): DataFrame containing 'uid' and 'iid' columns representing
                               user id and movie id, respectively.

        Returns:
            pd.DataFrame: DataFrame containing the estimated rating for the input user
                          and movie combination.
        '''
        uid = df['uid'].iloc[0]
        iid = df['iid'].iloc[0]
        prediction = self.model.predict(uid,iid)[3]
        return pd.DataFrame({'Estimated_Rating': [prediction]})
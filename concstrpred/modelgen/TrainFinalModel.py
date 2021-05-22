# Import pandas library
import pandas as pd

# Import pickle library
import pickle as pkl

# Import LightGBM library
import lightgbm as lgb

# Import warnings
import warnings
warnings.filterwarnings('ignore')

class TrainFinalModel():

    def train_lightgbm_final(self):

        # Load dataset
        df = pd.read_csv("data/output/Concrete_final.csv")

        # Prepare train data object specific to LGBM
        dataset = lgb.Dataset(df.drop('compressive_strength', axis = 1), label=df['compressive_strength'])

        # Train a LGBM model with Train object
        lgb_model = lgb.train(train_set=dataset, params={})

        # To persist the base model into hard-disk, uncomment the below line
        pkl.dump(lgb_model, open("model/lgb_final_model.mdl", 'wb'))

        print("LightGBM final model generated and persisted!!")

# Class ends here

def main():
    train_final_model = TrainFinalModel().train_lightgbm_final()

if __name__ == "__main__":
    main()

# import Flask class from the flask module
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas

# Create Flask object to run
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def home():
    return "Hi, Welcome to Flask!!"

########## LOADING OBJECTS. THIS WILL BE DONE ONLY ONCE ##########
print ("Loading model and objects...")
# Loading model
strength_pred_file = open('model/lgb_final_model.mdl', 'rb')
strength_pred_model = pickle.load(strength_pred_file)
strength_pred_file.close()

# Loading objects
feat_engg_file = open('model/feature_engg_cols.obj', 'rb')
feature_engg_cols = pickle.load(feat_engg_file)
feat_engg_file.close()
##################################################################

# Render Concrete mixture input page
@app.route('/input')
def input():
    return render_template('input.html')

# This function will be called when the input page is submitted
@app.route('/predict', methods=["POST"])
def predict():

    # Enter into this snippet of the code only if the method is POST.
    if request.method == "POST":

        # Get values from browser
        input_dict = request.form.to_dict()
        # {"cement": "400", "blast": "200"}

        # Extract keys and values from the input dictionary object
        form_keys =list(input_dict.keys())
        form_values =list(input_dict.values())

        # Convert them to float as they will be in String format
        form_values = map(float, form_values)

        # Construct the dictionary object with the existing keys and float values
        input_dict = dict(zip(form_keys, form_values))
        # {"cement": 400.0, "blast": 200.0}

        # Alternately, you can extract each field, cast to float, and pass it as a value
        # input_dict = dict{'cement': float(request.form['cement']),
        #               'blast': float(request.form['blast']),
        #               'flyash': float(request.form['flyash']),
        #               'water': float(request.form['water'])],
        #               'superplasticizer': float(request.form['superplasticizer']),
        #               'coarse_aggregate': float(request.form['coarse_aggregate']),
        #               'fine_aggregate': float(request.form['fine_aggregate']),
        #               'age': float(request.form['age'])
        #              }


        # Construct the dataframe out of the dictionary object
        input_df = pandas.DataFrame(input_dict, index=[0])
        print ("Input values: \n", input_df)

        # PERFORM ANY DATA TRANSFORMATION (Label and One-hot Encodings, dictionary mapping, etc)
        # AND NEW FEATURE GENERATION REQUIRED, OVER HERE ON THE input_df DATAFRAME.
        for col in feature_engg_cols:
            input_df[col] = input_df[col]/10


        # Pass the dataframe object to loaded ML model and do prediction
        yhat_strength = str(round(strength_pred_model.predict(input_df)[0], 2))
        print ("Predicted Concrete Strength: ", yhat_strength)

        return render_template('results.html', strength_predicted=yhat_strength)

#
# # Load the persisted ML model.
# # NOTE: The model will be loaded only once at the start of the server
# def load_model():
#     global concreteStrengthPredictorModel
#     print ("Loading model...")
#     concreteStrengthPredictorFile = open('model/concrete_lgbm_final_model_ver_1_0.sav', 'rb')
#     concreteStrengthPredictorModel = pickle.load(concreteStrengthPredictorFile)
#     concreteStrengthPredictorFile.close()


if __name__ == "__main__":
    print("**Starting Server...")

    # Call the function that loads Model
    # load_model()

    # Run Server
    app.run()

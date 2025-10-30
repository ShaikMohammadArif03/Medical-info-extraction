# Importing essential libraries and modules
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process

app = Flask(__name__)

# render home page
@app.route('/')
def home():
    title = 'medical prescription'
    return render_template('index.html', title=title) 

# render crop recommendation form page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'medical prescription'

    if request.method == 'POST':
        file = request.files.get('file')

        if file:
            img1 = file.read()
            with open('input.png', 'wb') as f:
                f.write(img1)

            from application_both_models import all_task
            prediction = all_task()

            # Load actual medicine names from the text file
            with open('medicine_names.txt', 'r') as file:
                actual_medicine_names = [line.strip() for line in file]

            # Predicted medicine names from your DL model
            predicted_medicine_names = prediction

            # Function to get the best match and similarity score
            def get_best_match(predicted_name, actual_names):
                matches = process.extractOne(predicted_name, actual_names, scorer=fuzz.token_sort_ratio)
                return matches

            # Replace old names with the most similar names
            improved_predictions = []
            for predicted_name in predicted_medicine_names:
                best_match, similarity_score = get_best_match(predicted_name, actual_medicine_names)
                improved_predictions.append(best_match)

            # Load the medicine data from the CSV files
            medicine_data = pd.read_csv('medicine_data.csv')
            medicine_data_full = pd.read_csv('medicine_data_full.csv')

            # Create an empty list to store the lowest price medicine details
            lowest_price_details = []
            full_details_list = []

            # Iterate through each medicine in the improved predictions list
            for medicine_name in improved_predictions:
                # Filter the medicine data for the specific medicine name
                filtered_data = medicine_data[medicine_data['medicine_names'] == medicine_name]
                
                if not filtered_data.empty:
                    lowest_price_row = filtered_data.loc[filtered_data['price'].idxmin()]
                    
                    # Append lowest price details
                    lowest_price_details.append({
                        'medicine_name': medicine_name,
                        'company_name': lowest_price_row['company_name'],
                        'price': lowest_price_row['price']
                    })

                    # Fetch additional details from full dataset
                    full_details_row = medicine_data_full[medicine_data_full['medicine_name'] == medicine_name]
                    if not full_details_row.empty:
                        full_details_list.append(full_details_row.to_dict(orient='records')[0])  # Convert to dictionary

            # Create DataFrames from lists of details
            lowest_price_details_df = pd.DataFrame(lowest_price_details)
            full_details_df = pd.DataFrame(full_details_list)

            return render_template(
                'disease-result.html',
                prediction="The Medicine Detected are As Follows:",
                lowest_price_details=lowest_price_details_df.to_html(classes='table table-striped'),
                full_details=full_details_df.to_html(classes='table table-striped'),
                title="Medical Prescription Detection"
            )

    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
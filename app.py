
#  customers are classified into clusters based on the scaled features of their demographics, financial status, and account information. The resulting clusters (Cluster column) in churn_with_clusters.csv can be used to tailor strategies for different customer segments.


# app.py
from flask import Flask, render_template, request
import numpy as np
from data_preprocessing import preprocess_data
from model_training import train_models
from dashboard import create_dashboard

app = Flask(__name__)

# Preprocess the data and train models
X_scaled, y, scaler, data, kmeans = preprocess_data('churn.csv')
best_classifier, classifiers = train_models(X_scaled, y)

# Integrate Dash app
dash_app = create_dashboard(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        credit_score = float(request.form['credit_score'])
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = int(request.form['num_of_products'])
        estimated_salary = float(request.form['estimated_salary'])
        has_credit_card = int(request.form.get('has_credit_card', 0))
        is_active_member = int(request.form.get('is_active_member', 0))
        
        # Prepare input data for prediction
        input_data = np.array([[credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, 0, 0, 1]])
        
        # Predict churn
        churn_prediction = best_classifier.predict(scaler.transform(input_data))
        
        # Predict customer segment
        customer_segment = kmeans.predict(scaler.transform(input_data))[0]
        
        offers = generate_retention_offers(input_data, customer_segment)
        return render_template('result.html', churn_prediction=churn_prediction[0], offers=offers)
    
    except Exception as e:
        return render_template('result.html', error_message=str(e))

def generate_retention_offers(input_data, customer_segment):
    offers = []
    credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, _, _, _ = input_data[0]

    # Offer based on credit score
    if credit_score < 600:
        offers.append('Offering credit score improvement program.')
    elif credit_score > 800:
        offers.append('Providing exclusive VIP banking services.')

    # Offer based on age
    if age > 60:
        offers.append('Special retirement planning services.')

    # Offer based on tenure and balance
    if tenure > 5 and balance > 50000:
        offers.append('Exclusive benefits for long-time customers with high balance.')

    # Offer based on number of products
    if num_of_products > 2:
        offers.append('Upgrade to premium account for free.')

    # Offer based on estimated salary
    if estimated_salary > 100000:
        offers.append('Personalized investment opportunities.')

    # Common offers
    if has_credit_card == 1:
        offers.append('Upgrade credit card with better rewards.')
    if is_active_member == 1:
        offers.append('Loyalty rewards for staying active.')

    # Offers based on customer segment
    if customer_segment == 0:
        offers.append('Special offers for new customers.')
    elif customer_segment == 1:
        offers.append('Incentives for long-term customers.')
    elif customer_segment == 2:
        offers.append('Rewards for high balance customers.')
    elif customer_segment == 3:
        offers.append('Exclusive deals for premium customers.')

    return offers

if __name__ == '__main__':
    app.run(debug=True)

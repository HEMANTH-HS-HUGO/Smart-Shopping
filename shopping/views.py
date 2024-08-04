from django.shortcuts import render
from django.http import HttpResponse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 

def index(request):
    return render(request, "shopping/index.html")

def load_data(request):
    shopping = pd.read_csv(r'G:\FULL STACK\Django\machinelearning\shopping\shopping.csv')

    # Store the dataset in the session so that it can be accessed by other functions
    request.session['shopping_data'] = shopping.to_dict(orient='records')

    context = {
        'dataset_head': shopping.head().to_html(index=False, classes='table table-striped table-hover')
    }
    return render(request, "shopping/index.html", context)  

def train_model(request):
    if request.method == 'POST':
        
        try:
            # shopping=pd.read_csv(r'G:\Semister 3.1\AiMl\DataSets\shopping.csv')
            shopping = pd.DataFrame(request.session['shopping_data'])
            le = LabelEncoder()
            shopping['Month'] = le.fit_transform(shopping['Month'])
            shopping['VisitorType'] = le.fit_transform(shopping['VisitorType'])
            shopping['Weekend'] = le.fit_transform(shopping['Weekend'])
            shopping['Revenue'] = le.fit_transform(shopping['Revenue'])
            X=shopping.iloc[:,:17]
            y=shopping.iloc[:,-1]
            X = preprocessing.StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
            knnmodel=KNeighborsClassifier(n_neighbors=2)
            knnmodel.fit(X_train,y_train)
            y_predict1=knnmodel.predict(X_test)
            accuracy=accuracy_score(y_test,y_predict1)
            recall = metrics.recall_score(y_test, y_predict1,zero_division=1)
            precision = metrics.precision_score(y_test, y_predict1,zero_division=1)
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Recall:, {recall:.2f}%")
            print(f"Precision:, {precision:.2f}%")

        except:
            print("Unable to Train the model")
    return render(request, 'shopping/index.html', {
        'accuracy': "{:.2f}".format(accuracy),
        'recall' : "{:.2f}".format(recall),
        'precision': "{:.2f}".format(precision),
        })

def predict_model(request):
    if request.method == 'POST':

        le = LabelEncoder()

        input_data = [
            int(request.POST['Administrative']),
            float(request.POST['Administrative_Duration']),
            int(request.POST['Informational']),
            float(request.POST['Informational_Duration']),
            int(request.POST['ProductRelated']),
            float(request.POST['ProductRelated_Duration']),
            float(request.POST['BounceRates']),
            float(request.POST['ExitRates']),
            float(request.POST['PageValues']),
            float(request.POST['SpecialDay']),
            le.fit_transform([request.POST['Month']])[0],
            int(request.POST['OperatingSystems']),
            int(request.POST['Browser']),
            int(request.POST['Region']),
            int(request.POST['TrafficType']),
            le.fit_transform([request.POST['VisitorType'] == 'Returning_Visitor'])[0],
            le.fit_transform([request.POST['Weekend'] == 'TRUE'])[0],
        ]

        print("Model is running")
        # Load the trained model
        shopping = pd.read_csv(r'G:\Semister 3.1\AiMl\DataSets\shopping.csv')
        le = LabelEncoder()
        shopping['Month'] = le.fit_transform(shopping['Month'])
        shopping['VisitorType'] = le.fit_transform(shopping['VisitorType'])
        shopping['Weekend'] = le.fit_transform(shopping['Weekend'])
        shopping['Revenue'] = le.fit_transform(shopping['Revenue'])
        
        X = shopping.iloc[:, :17]
        y = shopping.iloc[:, -1]
        X = preprocessing.StandardScaler().fit_transform(X)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=1)
        
        knnmodel = KNeighborsClassifier(n_neighbors=2)
        knnmodel.fit(X_train, y_train)

        # Make predictions using the input data
        prediction = knnmodel.predict([input_data])

        print(prediction)
        if prediction:
            print(prediction[0])
        else:
            print("Unable to Predict")

        return render(request, 'shopping/index.html', { 
            "prediction": prediction,
        })
    
    else:
        return render(request, 'shopping/index.html', {"prediction": "Unable to Predict" })




    
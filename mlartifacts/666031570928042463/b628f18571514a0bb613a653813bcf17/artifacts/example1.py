import mlflow
import mlflow.sklearn
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")


#Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

#Define the params for RF model
max_depth = 5
n_estimator = 9

#Mention your experiment below
mlflow.set_experiment('Example_experiments')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimator,random_state=26) 
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric("Accuracy",accuracy)
    mlflow.log_param("Max_depth",max_depth)
    mlflow.log_param("n_estimator",n_estimator)

    #Creating a confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    #save plot
    plt.savefig("Confusion_matrix.png")

    # log artifacts using mlfow
    mlflow.log_artifact("Confusion_matrix.png")
    mlflow.log_artifact(__file__)

    #tags
    mlflow.set_tags({"Author":"Ganesh","Project":"Wine CLassification"})

    # Log the model
    mlflow.sklearn.log_model(rf,"Random-Forest-Model")

    print("Accuracy",accuracy)
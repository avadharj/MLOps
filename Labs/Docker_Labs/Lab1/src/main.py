# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == '__main__':
    # Load the Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an SVM classifier
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model and scaler to files
    joblib.dump(model, 'wine_model.pkl')
    joblib.dump(scaler, 'wine_scaler.pkl')

    print("The model training was successful")
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def run_model(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    y_train = train_df['median_house_value']
    X_train = train_df.drop(columns=['median_house_value'])
    
    if 'median_house_value' in test_df.columns:
        X_test = test_df.drop(columns=['median_house_value'])
    else:
        X_test = test_df.copy()

    cols_to_log = ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    for col in cols_to_log:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)
    
    model = DecisionTreeClassifier(max_depth=20, min_samples_leaf=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    
    with open("S6520241.txt", "w") as fh:
        for pred in predictions:
            fh.write(f"{pred}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    args = parser.parse_args()
    
    run_model(args.train, args.test)
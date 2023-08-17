import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
from sklearn.neighbors import NearestNeighbors

def find_potential_outcomes(H, observed_output, sensitive_feature): #H is client_data

    G = H.copy(deep=True)
    G.reset_index(inplace=True)
    G = G.apply(pd.to_numeric)
    #Z is samples without output
    #now make output y as dataframe and connect it with the samples


    psm = PsmPy(G, treatment=sensitive_feature, indx='index', exclude = [])
    psm.logistic_ps(balance = True)
    psm.predicted_data
    psm.knn_matched(matcher='propensity_score', replacement=True, caliper=None)
    
    psm.matched_ids
    psm.predicted_data['propensity_logit']
    ps = psm.predicted_data['propensity_logit']
    #print(prop_logit)



    caliper = np.std(ps) * 0.5
    #print(f'caliper (radius) is: {caliper:.4f}')

    n_neighbors = 2

    # setup knn
    knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)
    
    potential_output = find_potential_outcomes_ordered(psm, observed_output, knn, sensitive_feature)
    return potential_output
    
def find_potential_outcomes_ordered(psm, observed_output, knn, sensitive_feature):
    female = psm.predicted_data[psm.predicted_data[sensitive_feature] == 1]
    male = psm.predicted_data[psm.predicted_data[sensitive_feature] == 0]

    knn.fit(male[['propensity_logit']])
    _, neighbor_indexes_female = knn.kneighbors(female[['propensity_logit']])

    knn.fit(female[['propensity_logit']])
    _, neighbor_indexes_male = knn.kneighbors(male[['propensity_logit']])

    # Create dictionaries with keys as original indexes and values as potential outcomes
    potential_output_female_dict = {female.index[i]: observed_output[idx] for i, idx in enumerate(neighbor_indexes_female[:, 0])}
    potential_output_male_dict = {male.index[i]: observed_output[idx] for i, idx in enumerate(neighbor_indexes_male[:, 0])}

    # Combine the dictionaries
    potential_output_dict = {**potential_output_female_dict, **potential_output_male_dict}

    # Create a list of potential outcomes, maintaining the original order in the dataset
    potential_output_ordered = [potential_output_dict[i] for i in sorted(potential_output_dict.keys())]

    return potential_output_ordered

def load_adult(url):
    data = pd.read_csv(url)
    #data = shuffle(data)
   

    # Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('income', axis=1)
    y = LabelEncoder().fit_transform(data['income'])
    
    return X, y

def load_adult_age_country(url, sensitive_feature):
    data = pd.read_csv(url)
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['age','fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    data = train_df
    mask = (data['native.country'] =='United-States') |(data['native.country'] =='Cuba') | (data['native.country'] =='Jamaica') | (data['native.country'] =='Mexico') | (data['native.country'] =='Puerto-Rico') | (data['native.country'] =='Honduras') | (data['native.country'] =='Canada') | (data['native.country'] =='Haiti') | (data['native.country'] =='Dominican-Republic') | (data['native.country'] =='El-Salvador') | (data['native.country'] ==' Guatemala') | (data['native.country'] ==' Outlying-US(Guam-USVI-etc)') | (data['native.country'] =='Nicaragua') | (data['native.country'] =='Columbia') | (data['native.country'] =='Ecuador') | (data['native.country'] =='Peru') | (data['native.country'] =='Trinadad|Tobago') 
    df1 = data[mask]
    
    mask = (data['native.country'] =='India') | (data['native.country'] =='?') | (data['native.country'] =='South')| (data['native.country'] =='Iran')| (data['native.country'] =='Philippines')| (data['native.country'] =='Cambodia')| (data['native.country'] =='Thailand')| (data['native.country'] =='Laos')| (data['native.country'] =='Taiwan')| (data['native.country'] =='China')| (data['native.country'] =='Japan')| (data['native.country'] =='Vietnam')| (data['native.country'] =='Hong') 
    df2 = data[mask]
    
    mask = (data['native.country'] =='England') | (data['native.country'] =='Germany')| (data['native.country'] =='Italy')| (data['native.country'] =='Poland')| (data['native.country'] =='Portugal')| (data['native.country'] =='France')| (data['native.country'] =='Yugoslavia')| (data['native.country'] =='Scotland')| (data['native.country'] =='Greece')| (data['native.country'] =='Ireland')| (data['native.country'] =='Hungary')| (data['native.country'] =='Holand-Netherlands')
    df3 = data[mask]
    
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    
    
    categorical_columns = ['native.country']
    for col in categorical_columns:
        encoder = LabelEncoder()
        df1[col] = encoder.fit_transform(df1[col])
        df2[col] = encoder.fit_transform(df2[col])
        df3[col] = encoder.fit_transform(df3[col])
        test_df[col] = encoder.fit_transform(test_df[col])
   
    #scalrising the 'age' column as well
    #numerical_columns = ['age']
    #scaler = StandardScaler()
    #df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    #df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    #df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    # Split the data into features and labels
    
    X_client1 = df1.drop('income', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['income'])
    
    X_client2 = df2.drop('income', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['income'])
    
    X_client3 = df3.drop('income', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['income'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
   
    
    
    
    
    s_client2 = X_client2[sensitive_feature]
    y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('income', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['income'])
    sex_column = X_test['sex']
    column_names_list = X_test.columns.tolist()
    ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

def load_adult_age(url, sensitive_feature):
    data = pd.read_csv(url)
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        
    # Normalize numerical columns
    numerical_columns = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['age'] >=0) & (data['age'] <=29)
    df1 = data[mask]
    mask = (data['age'] >=30) & (data['age'] <=39)
    df2 = data[mask]
    mask = (data['age'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['age']
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    # Split the data into features and labels
    test_df = result = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_client1 = df1.drop('income', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['income'])
    
    X_client2 = df2.drop('income', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['income'])
    
    X_client3 = df3.drop('income', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['income'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
   
    
    
    
    
    s_client2 = X_client2[sensitive_feature]
    y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('income', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['income'])
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

def load_bank(url):
    data = pd.read_csv(url)
    #data = shuffle(data)
   

    # Encode categorical columns
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'previous', 'poutcome']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('y', axis=1)
    y = LabelEncoder().fit_transform(data['y'])
    #print("bismillah")
    return X, y

def load_bank_age(url, sensitive_feature):
    data = pd.read_csv(url)
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'previous', 'poutcome']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['balance', 'day', 'duration', 'campaign', 'pdays']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['age'] >=0) & (data['age'] <=29)
    df1 = data[mask]
    mask = (data['age'] >=30) & (data['age'] <=39)
    df2 = data[mask]
    mask = (data['age'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['age']
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    # Split the data into features and labels
    test_df = result = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_client1 = df1.drop('y', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['y'])
    
    X_client2 = df2.drop('y', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['y'])
    
    X_client3 = df3.drop('y', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['y'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
   
    
    
    
    
    s_client2 = X_client2[sensitive_feature]
    y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('y', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['y'])
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential


def load_default(url):
    data = pd.read_csv(url)
    #data = shuffle(data)
    # Encode categorical columns
    categorical_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('y', axis=1)
    y = LabelEncoder().fit_transform(data['y'])
    #print("bismillah")
    return X, y

def load_default_age(url, sensitive_feature):
    data = pd.read_csv(url)
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['AGE'] >=0) & (data['AGE'] <=29)
    df1 = data[mask]
    mask = (data['AGE'] >=30) & (data['AGE'] <=39)
    df2 = data[mask]
    mask = (data['AGE'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['AGE']
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    # Split the data into features and labels
    test_df = result = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_client1 = df1.drop('y', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['y'])
    
    X_client2 = df2.drop('y', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['y'])
    
    X_client3 = df3.drop('y', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['y'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
   
    
    
    
    
    s_client2 = X_client2[sensitive_feature]
    y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('y', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['y'])
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
def load_law(url):
    data = pd.read_csv(url)
    data['y'] = data['y'].replace({0: 1, 1: 0})
    data['sex'] = data['sex'].replace({0: 1, 1: 0})
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ["decile1b", "decile3", "fulltime", "fam_inc", "sex", "race", "tier"]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ["lsat", "ugpa", "zfygpa", "zgpa"]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('y', axis=1)
    y = LabelEncoder().fit_transform(data['y'])
    #print("bismillah")
    return X, y

def load_law_income(url, sensitive_feature):
    data = pd.read_csv(url)
    data['y'] = data['y'].replace({0: 1, 1: 0})
    data['sex'] = data['sex'].replace({0: 1, 1: 0})
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ["decile1b", "decile3", "fulltime", "fam_inc", "sex", "race", "tier"]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ["lsat", "ugpa", "zfygpa", "zgpa"]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    data = train_df
    
    mask = (data['fam_inc'] >=1) & (data['fam_inc'] <=2)
    df1 = data[mask]
    mask = (data['fam_inc'] ==3)
    df2 = data[mask]
    mask = (data['fam_inc'] >=4) & (data['fam_inc'] <=5)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    # Split the data into features and labels
    
    X_client1 = df1.drop('y', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['y'])
    
    X_client2 = df2.drop('y', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['y'])
    
    X_client3 = df3.drop('y', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['y'])
    
    
    s_client1 = X_client1[sensitive_feature]
    y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
    
    
    s_client2 = X_client2[sensitive_feature]
    y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('y', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['y'])
    sex_column = X_test['sex']
    column_names_list = X_test.columns.tolist()
    ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

def load_kdd(url):
    data = pd.read_csv(url)
    data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ["class-of-worker", "education", "enroll-in-edu-inst-last-wk", "marital-stat", "major-industry-code", "major-occupation-code", "race", "hispanic-origin", "sex", "member-of-a-labor-union", "reason-for-unemployment", "full-or-part-time-employment-stat", "tax-filer-stat","region-of-previous-residence", "state-of-previous-residence", "detailed-household-and-family-stat","detailed-household-summary-in-household", "migration-code-change-in-msa", "migration-code-change-in-reg","migration-code-move-within-reg", "live-in-this-house-1-year-ago", "migration-prev-res-in-sunbelt", "family-members-under-18", "country-of-birth-father", "country-of-birth-mother", "country-of-birth-self", "citizenship", "own-business-or-self-employed", "fill-inc-questionnaire-for-veterans-admin"]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ["age", "detailed-industry-recode","detailed-occupation-recode","wage-per-hour","capital-gains", "capital-losses", "num-persons-worked-for-employer", "dividends-from-stocks", "veterans-benefits","weeks-worked-in-year", "year"]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('class', axis=1)
    y = LabelEncoder().fit_transform(data['class'])
    #print("bismillah")
    return X, y

def load_kdd_age(url, sensitive_feature):
    data = pd.read_csv(url)
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ["class-of-worker", "education", "enroll-in-edu-inst-last-wk", "marital-stat", "major-industry-code", "major-occupation-code", "race", "hispanic-origin", "sex", "member-of-a-labor-union", "reason-for-unemployment", "full-or-part-time-employment-stat", "tax-filer-stat","region-of-previous-residence", "state-of-previous-residence", "detailed-household-and-family-stat","detailed-household-summary-in-household", "migration-code-change-in-msa", "migration-code-change-in-reg","migration-code-move-within-reg", "live-in-this-house-1-year-ago", "migration-prev-res-in-sunbelt", "family-members-under-18", "country-of-birth-father", "country-of-birth-mother", "country-of-birth-self", "citizenship", "own-business-or-self-employed", "fill-inc-questionnaire-for-veterans-admin"]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ["detailed-industry-recode","detailed-occupation-recode","wage-per-hour","capital-gains", "capital-losses", "num-persons-worked-for-employer", "dividends-from-stocks", "veterans-benefits","weeks-worked-in-year", "year"]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['age'] >=0) & (data['age'] <=29)
    df1 = data[mask]
    mask = (data['age'] >=30) & (data['age'] <=39)
    df2 = data[mask]
    mask = (data['age'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['age']
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    # Split the data into features and labels
    test_df = result = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_client1 = df1.drop('class', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['class'])
    
    X_client2 = df2.drop('class', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['class'])
    
    X_client3 = df3.drop('class', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['class'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
    
    
    s_client2 = X_client2[sensitive_feature]
    y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('class', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['class'])
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
def load_dataset(url, dataset_name, num_clients, sensitive_feature, distribution_type):
    if dataset_name == 'adult':
        if distribution_type == 'random':
            X, y = load_adult(url)
        else:
            data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_adult_age(url, sensitive_feature)
            return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    
    elif dataset_name == 'bank':
        if distribution_type == 'random':
            X, y = load_bank(url)
        else:
            data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_bank_age(url, sensitive_feature)
            return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    
    elif dataset_name == 'default':
        if distribution_type == 'random':
            X, y = load_default(url)
        else:
            data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_default_age(url, sensitive_feature)
            return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
            
    elif dataset_name == 'law':
        if distribution_type == 'random':
            X, y = load_law(url)
        else:
            data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_law_income(url, sensitive_feature)
            return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    
    elif dataset_name == 'kdd':
        if distribution_type == 'random':
            X, y = load_kdd(url)
        else:
            data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_kdd_age(url, sensitive_feature)
            return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
        
    k = 0
    if k==0:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        sex_column = X_test[sensitive_feature]
        column_names_list = X_temp.columns.tolist()
        # Convert the pandas Series to a Python list
        sex_list = sex_column.tolist()
        data_dict = {}
        
        for i in range(num_clients):
            if i == num_clients - 1:
                X_client, y_client = X_temp, y_temp
            else:
                X_temp, X_client, y_temp, y_client = train_test_split(X_temp, y_temp, test_size=1/(num_clients-i), random_state=42)
            
            s_client = X_client[sensitive_feature]
            #compute potential outcomes
            y_potential_client = find_potential_outcomes(X_client,y_client, sensitive_feature)
            # Convert to PyTorch tensors
            X_client = torch.tensor(X_client.values, dtype=torch.float32)
            y_client = torch.tensor(y_client, dtype=torch.float32)
            s_client = torch.from_numpy(s_client.values).float()
            y_potential_client = torch.tensor(y_potential_client, dtype=torch.float32)
            
            # Store the client data in the dictionary
            data_dict[f"client_{i+1}"] = {"X": X_client, "y": y_client, "s": s_client, "y_pot": y_potential_client}
        ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
        ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
       
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

def get_data(client_name, data_dict):
    client_data = data_dict.get(client_name, {})
    X = client_data.get("X")
    y = client_data.get("y")
    s = client_data.get("s")
    y_pot = client_data.get("y_pot")
    return X, y, s, y_pot

 
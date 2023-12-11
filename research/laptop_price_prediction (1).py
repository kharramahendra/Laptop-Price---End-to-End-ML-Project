import numpy as np
import matplotlib.pyplot as plt

# !pip install plotly.express

import plotly.express as pex

import pandas as pd
import seaborn as sns

df = pd.read_csv('data.csv')

df.head(2)

"""# **Part-1 EDA, ANALYSIS, ADVANCED MISSING VALUES IMPUTATION,AND NEXT LEVEL PRE PROCESSING **"""

df['CPU'].value_counts()

sns.barplot(df,x='Ram', y='price')

sns.barplot(df,x='ROM_type',y='price')

plt.figure(figsize=(16,4))
sns.barplot(df,x='brand',y='price')

# plt.figure(figsize=(16,9))
pex.bar(x=df['CPU'].value_counts().index,y=df['CPU'].value_counts().values,template='plotly_dark')

pex.bar(df,x='CPU',color='processor')

pex.histogram(df,x='spec_rating')

fig = pex.scatter(df,x='resolution_width',y='resolution_height',color='OS',template='plotly_dark')
fig.show()

df.head(2)

df['Ram'].value_counts()

list(df['processor'].value_counts().index)

import re

def parse_processor_name(processor_name):
    # Define regular expressions for extracting information
    regexes = [
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel|AMD) (Core|i\d+|Celeron|Pentium|Atom|Ryzen|Athlon) ?(\w*)'),
        re.compile(r'(Apple) (M1|M2(?: Pro)?(?: Max)?)'),
        re.compile(r'(Intel) (Celeron|Pentium|Atom) (\w+)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Celeron) (\w+)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Pentium) (\w+)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Core) (i\d+) (\w*)'),
        re.compile(r'(\d+)(?:th|rd|st) Gen (Intel) (Core) (i\d+)'),
    ]

    # Match the regular expressions against the processor name
    for regex in regexes:
        match = regex.match(processor_name)
        if match:
            groups = match.groups()
            if groups[0] == 'Apple':
                return {'generation':'1','company': groups[0],'model_type': 'M1', 'version': groups[1]}
            elif groups[0] == 'Intel':
                if groups[2] in ['Celeron', 'Pentium', 'Atom']:
                    return {'generation': groups[1], 'company': groups[0], 'model_type': groups[2], 'version': groups[3]}
                elif groups[2] == 'Core':
                    return {'generation': groups[1], 'company': groups[0], 'model_type': f'{groups[2]} {groups[4]}', 'version': groups[5]}
                else:
                    return None
            else:
                return {'generation': groups[0], 'company': groups[1], 'model_type': groups[2], 'version': groups[3]}

    return None

processors = list(df['processor'])
new = []
for processor in processors:
  print(processor)
  value = parse_processor_name(processor)
  new.append(value)

new.count(None)

new[0]

processor_data = []
for obj in new:
  if obj is None:
    processor_data.append([None,None,None,None])
  else:
    processor_data.append([obj['company'],obj['generation'],obj['version'],obj['model_type']])

len(processor_data)

prodf = pd.DataFrame(processor_data,columns=['company','generation','version','model_type'])

prodf

df.head(2)

list(df['GPU'].value_counts().index)

import re

def get_gpu_type(gpu_name):
    # Define regular expressions for extracting GPU type information
    regexes = [
        re.compile(r'(NVIDIA|AMD)\s*(Radeon)?'),
        re.compile(r'(Apple)\s*(Integrated Graphics)'),
        re.compile(r'(Intel)\s*(Iris Xe Graphics|UHD Graphics|HD Graphics|Graphics)?'),
        re.compile(r'(ARM)\s*(Mali G\d+)'),
    ]

    # Match the regular expressions against the GPU name
    for regex in regexes:
        match = regex.search(gpu_name)
        if match:
            groups = match.groups()
            gpu_type = groups[1] if len(groups) > 1 and groups[1] else groups[0] if groups[0] else None
            return gpu_type

    return None

get_gpu_type('4GB AMD Radeon RX 6500M')

gpus = list(df['GPU'])
gpu_data_list = []
for gpu in gpus:
  # print(gpu)
  value = get_gpu_type(gpu)
  gpu_data_list.append(value)

gpu_data_list.count(None)

gpu_data_list

df['gpu_type'] = gpu_data_list

# gpu data is too complex and can't be handled easily. for that I am just taking brand name of the gpu as new column.

df.head(3)

df['gpu_type'].value_counts()

plt.figure(figsize=(16,6))
sns.barplot(df,x='gpu_type',y='price')

display_size_options = list(df['display_size'].value_counts())

display_size_options

list(df['CPU'].value_counts().index)

pex.bar(df,x='CPU',y='price',template='plotly_dark')

import re

def extract_cores_threads(cpu_name):
    # Check for the presence of Cores and Threads in the name
    cores_match = re.search(r'(\d+|Dual|Quad|Hexa|Octa)\s*Cores?', cpu_name)
    threads_match = re.search(r'(\d+)\s*Threads?', cpu_name)

    # Extract the number of cores and threads from the matches
    cores = 0 if cores_match is None else cores_match.group(1)
    threads = 0 if threads_match is None else threads_match.group(1)

    # Convert 'Dual', 'Quad', 'Hexa', 'Octa' to corresponding numbers
    cores_dict = {'Dual': 2, 'Quad': 4, 'Hexa': 6, 'Octa': 8}
    cores = cores_dict.get(cores, cores)

    return int(cores), int(threads)

print(extract_cores_threads("Octa Core (4P + 4E)"))

cpu_list = list(df['CPU'])

cpu_data = []
for cpu in cpu_list:
  cpu_data.append(extract_cores_threads(cpu))

cpu_data

df[['cpu_core','cpu_threads']] = cpu_data

df

# Now we have hanled 3 main columns - Processor, CPU and GPU

processor_data
df[['processor_brand','processor_gen','processor_version','processor_model']] = processor_data

df.head(1)

df.columns

data = df.drop(['Unnamed: 0.1', 'Unnamed: 0','name','processor','CPU','Ram_type','GPU','processor_model'],axis=1)

data.head(2)

data.isnull().sum()

# Lets make custom encoders for every categorical variale

data.select_dtypes(include=['int64','float64'])

data.select_dtypes(include=['object'])

# brand,os,gpu_type,processor_brand -> label encoder

data.update(data['Ram'].apply(lambda x: int(x.split('GB')[0])))

data['Ram'].value_counts()

data.update(data['ROM'].apply(lambda x: int(x.split('GB')[0]) if 'GB' in x else int(x.split('TB')[0])*1024))

data['ROM'].value_counts()

data.update(data['ROM_type'].apply(lambda x: 1 if 'SSD' in x else 0))

data['ROM_type'].value_counts()

# lets fill mode values where processor_gen is None.
# I am filling mode coz, mean and meadian can be any continous value or a value which is not suitable for generation or also for version

data.update(data['processor_gen'].fillna(data['processor_gen'].mode()[0],inplace=True))

data['processor_gen'].isnull().sum()

# all processor comanies have different version coding so we need to handle this column according to their company

data.update(data['processor_brand'].fillna(data['processor_brand'].mode()[0],inplace=True))

for brand in data['processor_brand'].value_counts().index:
  print(brand)
  data.update(data[data['processor_brand']==brand]['processor_version'].replace(np.nan,data[data['processor_brand']==brand]['processor_version'].mode()[0]))

data['processor_version'].value_counts()

data.isnull().sum()

data['gpu_type'].fillna(data['gpu_type'].mode()[0],inplace=True)

# data = data.drop('processor_model',axis=1)

data.head()

data.isnull().sum()

data[['Ram','ROM','ROM_type','processor_gen']] = data[['Ram','ROM','ROM_type','processor_gen']].apply(np.int64)

data.select_dtypes(include=['object'])

from sklearn.preprocessing import LabelEncoder

data['OS'].value_counts()

data.update(data['OS'].replace('Windows 11  OS','Windows 11 OS'))
data.update(data['OS'].replace('Windows 10  OS','Windows 10 OS'))

data['OS'].value_counts()

# brand_le = LabelEncoder()
# data['brand'] = brand_le.fit_transform(data['brand'])

# gpu_le = LabelEncoder()
# data['gpu_type'] = gpu_le.fit_transform(data['gpu_type'])

# processor_brand_le = LabelEncoder()
# data['processor_brand'] = processor_brand_le.fit_transform(data['processor_brand'])

# processor_version_le = LabelEncoder()
# data['processor_version'] = processor_version_le.fit_transform(data['processor_version'])

# os_le = LabelEncoder()
# data['OS'] = gpu_le.fit_transform(data['OS'])

data.head()

data.info()

# now data is ready to split into train and test data and ready for scaling to develop ml project

pex.imshow(data.corr(),template='plotly_dark')

# preprocessor for production deployment

# 1. first of all we apply functions on preprocessor,cpu and add some columns
# 2. remove all extra columns
# 3. then apply different functionalities for every column, missing values imputation,converting into int64 ,some more
# 4. categorical encoding using LabelEncoder

# maximum pre processing has complex tasks and also we cant use labelencoder in pipelines

encoder = LabelEncoder()

data['brand'].value_counts()

data[['brand', 'OS', 'gpu_type', 'processor_brand', 'processor_version']].apply(encoder.fit_transform)

data[['brand', 'OS', 'gpu_type', 'processor_brand', 'processor_version']].iloc[0]

"""# **Part-2 Preprocessor with encoders and scaler for new data **"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

cat_cols = ['brand', 'OS', 'gpu_type', 'processor_brand', 'processor_version']
cat_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(sparse=False,drop='first',handle_unknown='ignore')),
        ("scaler", StandardScaler()),
    ]
)

num_cols = [ 'spec_rating', 'Ram', 'ROM', 'ROM_type', 'display_size',
       'resolution_width', 'resolution_height', 'warranty', 'cpu_core',
       'cpu_threads', 'processor_gen']
num_cat_transformer = Pipeline(
    steps=[
        # ("encoder", LabelEncoder()),
        ("scaler", StandardScaler()),
    ]
)

transformer = ColumnTransformer(
    transformers=[
        ('categorical_transformer',cat_transformer,cat_cols),
        ("numerical_transformer",num_cat_transformer,num_cols)
    ]
)

from sklearn.model_selection import train_test_split

X = data.drop('price',axis=1)
y = data['price']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

new_x_train = transformer.fit_transform(x_train)

new_x_train.shape

new_x_test = transformer.transform(x_test)

"""## **Part-3 Machine learning Algorithms**"""

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

model = Ridge()

parameters = {
    # 'loss':['log_loss','exponential'],
    # 'penalty':['l1', 'l2', 'elasticnet', None],
    "criterion":['gini','entropy','log_loss'],
    # "criterion":['friedman_mse','squared_error'],
    "max_depth":[1,2,3,4,5,6,7,8,9,10],
    "oob_score":[True,False]
}

from sklearn.model_selection import GridSearchCV

parameters = {'alpha':[1,2,3,4,5,10,20,30,40,50,70]}
ridge_cv = GridSearchCV(model,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_cv.fit(new_x_train,y_train)

ridge_cv.best_params_

ridge_cv.best_score_

ridge_pred = ridge_cv.predict(new_x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test,ridge_pred)
score

lasso = Lasso()

parameters = {'alpha':[1,2,3,4,5,10,20,30,40,50,70]}
lasso_cv = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_cv.fit(new_x_train,y_train)

lasso_pred = lasso_cv.predict(new_x_test)

r2_score(y_test,lasso_pred)

# dicision tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeRegressor()
dt_parameters = {
    "criterion": ["mse", "mae", "friedman_mse"],
    "splitter": ["best", "random"],
    "max_depth": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
dt_cv = GridSearchCV(dt, dt_parameters, scoring='neg_mean_squared_error', cv=5)
dt_cv.fit(new_x_train, y_train)

dt_cv.best_params_

dt_cv.best_score_

dt_pred = dt_cv.predict(new_x_test)

r2_score(y_test,dt_pred)

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

svm = SVR()
svm_parameters = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.1, 0.01, 0.001],
}
svm_cv = GridSearchCV(svm, svm_parameters, scoring='neg_mean_squared_error', cv=5)
svm_cv.fit(new_x_train, y_train)

svm_preds = svm_cv.predict(new_x_test)

r2_score(y_test,svm_preds)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

gb = GradientBoostingRegressor()
gb_parameters = {
    "loss": ["ls", "lad", "huber", "quantile"],
    "learning_rate": [0.001, 0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 4, 5, 6],
}
gb_cv = GridSearchCV(gb, gb_parameters, scoring='neg_mean_squared_error', cv=5)
gb_cv.fit(new_x_train, y_train)

gb_preds = gb_cv.predict(new_x_test)

r2_score(y_test,gb_preds)

# KNN

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

knn = KNeighborsRegressor()
knn_parameters = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2],
}
knn_cv = GridSearchCV(knn, knn_parameters, scoring='neg_mean_squared_error', cv=5)
knn_cv.fit(new_x_train, y_train)

knn_preds = knn_cv.predict(new_x_test)

r2_score(y_test,knn_preds)

# random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()
rf_parameters = {
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "oob_score": [True, False],
}
rf_cv = GridSearchCV(rf, rf_parameters, scoring='neg_mean_squared_error', cv=5)
rf_cv.fit(new_x_train, y_train)

rf_preds = rf_cv.predict(new_x_test)

r2_score(y_test,rf_preds)



#   KNN is giving best accuracy from all these algorithsm. So let's train the KNN with best parameters.

knn_cv.best_params_

knn_final = KNeighborsRegressor(algorithm="kd_tree",n_neighbors=7,weights='distance')

knn_final.fit(new_x_train,y_train)

final_preds = knn_final.predict(new_x_test)

r2_score(y_test,final_preds)

import joblib

joblib.dump(knn_final, 'model.pkl')

model = joblib.load('model.pkl')

new_data = transformer.transform(x_test.head(5))

y_test

x_test.head(5)

model.predict(new_data)

# model is giving almost 80 % accuracy. Laptop prices may vary with offers and all so remaining 20% can me covered as model is predicting slightly less prices


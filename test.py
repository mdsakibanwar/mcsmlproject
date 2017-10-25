# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "adult.data"
names = ['age', 'workclass', 'fnlwgt' , 'education', 'education-num', 'marital-status','occupation','relatioship','race','sex','capital-gain','capital-loss','hours-per-week', 'native-country','class']
dataset = pandas.read_csv(url, names=names)


dataset.isnull().sum()


# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

gnb = GaussianNB()
arr = dataset.values
X = arr[:,0:13]
Y = arr[:,14]

gnb.fit(X,Y)

predict = gnb.predict(['25', 'Private', '226802', '11th', '7', 'Never-married', 'Machine-op-inspct', 'Own-child', 'Black', 'Male', '0', '0', '40', 'United-States'])

print predict



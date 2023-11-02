import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from termcolor import colored as cl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

rcParams['figure.figsize'] = (25, 20)

# Load the diabetes dataset (replace 'diabetes.csv' with your dataset file)
df = pd.read_csv('diabetes.csv')

# Remove any unnecessary columns
# df.drop('Unnamed: 0', axis=1, inplace=True)  # Uncomment this line if your dataset contains an unnecessary index column

print(cl(df.head(), attrs=['bold']))

df.info()

# Define your independent and dependent variables
X_var = df.drop('Outcome', axis=1).values  # independent variables
y_var = df['Outcome'].values  # dependent variable

print(cl('X variable samples : {}'.format(X_var[:5]), attrs=['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs=['bold']))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=0)

print(cl('X_train shape : {}'.format(X_train.shape), attrs=['bold'], color='red'))
print(cl('X_test shape : {}'.format(X_test.shape), attrs=['bold'], color='red'))
print(cl('y_train shape : {}'.format(y_train.shape), attrs=['bold'], color='green'))
print(cl('y_test shape : {}'.format(y_test.shape), attrs=['bold'], color='green'))

# Create a Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(X_train, y_train)

# Make predictions on the test set
pred_model = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, pred_model)
print(cl('Accuracy of the model is {:.2%}'.format(accuracy), attrs=['bold']))

# Plot the decision tree
plt.figure(figsize=(20, 15))
plot_tree(model, filled=True, rounded=True, feature_names=df.columns[:-1], class_names=['0', '1'])
plt.savefig('tree_visualization.png')
# Import necessary libraries

# Create and fit your decision tree model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)  # Replace X and y with your data

# Create a text representation of the decision tree
tree_text = export_text(clf, feature_names=feature_names)  # Replace feature_names

# Print or save the text representation to a file
print(tree_text)

# Create a visualization of the decision tree using matplotlib
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)  # Replace feature_names and class_names
plt.savefig("tree_visualization.png")  # Save the visualization as an image file
img = mpimg.imread("tree_visualization.png")

# Display the visualization
plt.imshow(img)
plt.show()

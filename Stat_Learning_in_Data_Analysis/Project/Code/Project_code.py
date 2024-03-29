# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
from keras.utils.vis_utils import plot_model

from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

np.random.seed(20221107)

# %%
# Load the data
path = "../data/weatherAUS.csv"

# Assign format to the date column
data = pd.read_csv(
    path,
    dtype={
        "Location": "category",
        "WindGustDir": "category",
        "WindDir9am": "category",
        "WindDir3pm": "category",
        "RainToday": "category",
    },
)

# Convert to datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Drop the rows with missing values
data = data.dropna(
    axis=0
)

# Replace Yes and No with 1 and 0
data["RainTomorrow"] = data["RainTomorrow"].replace(
    "Yes", 1
)
data["RainTomorrow"] = data["RainTomorrow"].replace(
    "No", 0
)

name = data.columns

for j in name:
    if data[j].dtypes == "category":
        # Get the Dummy Variables for the categorical variables
        dummy = pd.get_dummies(data[j], prefix=j)
        data = pd.concat([data, dummy], axis=1)
        del data[j]

x = data.iloc[:, 1:]
del x["RainTomorrow"]


x_columns = x.columns

# Label
y = data["RainTomorrow"]

# print(y.value_counts())
# print(y.value_counts() / len(y))

ros = RandomUnderSampler(random_state=0)
x, y = ros.fit_resample(x, y)

print(y.value_counts())
print(y.value_counts() / len(y))

# If you want to map the data to (0,1), uncomment the following two lines

Mm = MinMaxScaler()
x = Mm.fit_transform(x)

# If you want to use PCA, uncomment the following two lines
# If you don't use PCA, SVM will consume a long time

# pca = PCA(n_components=0.95)
# x = pca.fit_transform(x)

print(len(x[0]))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
font1 = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": 16,
}

# %%
# Logistic Regression
logreg = LogisticRegression(max_iter=3000)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print(
    "Test Accuracy:",
    metrics.accuracy_score(y_test, y_pred),
    "\nTrain Accuracy: ",
    metrics.accuracy_score(y_train, logreg.predict(x_train)),
)

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g", cbar=False)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.ylabel("Actual label", font1)
plt.xlabel("Predicted label", font1)


plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]

plt.savefig("log_cm_undersampling.eps", bbox_inches="tight")

plt.show()

# Classification Report
target_names = ["Not Raining", "Raining"]
print(classification_report(y_test, y_pred, target_names=target_names))

# %%
for i in np.argsort(np.abs(logreg.coef_[0]))[-1:-11:-1]: 
    print(x_columns[i], logreg.coef_[0][i], np.abs(logreg.coef_[0][i]))

# %%
# SVM
clf = svm.SVC(kernel="linear")

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(
    "Test Accuracy:",
    metrics.accuracy_score(y_test, y_pred),
    "\nTrain Accuracy: ",
    metrics.accuracy_score(y_train, clf.predict(x_train)),
)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g", cbar=False)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.ylabel("Actual label", font1)
plt.xlabel("Predicted label", font1)


plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]

plt.savefig("svm_cm_undersampling_pca.eps", bbox_inches="tight")

plt.show()

target_names = ["Not Raining", "Raining"]
print(classification_report(y_test, y_pred, target_names=target_names))

# %%
# SVM with polynomial kernel
clf = svm.SVC(kernel="poly")

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(
    "Test Accuracy:",
    metrics.accuracy_score(y_test, y_pred),
    "\nTrain Accuracy: ",
    metrics.accuracy_score(y_train, clf.predict(x_train)),
)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g", cbar=False)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.ylabel("Actual label", font1)
plt.xlabel("Predicted label", font1)


plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]

plt.savefig("svm_cm_undersampling_poly_pca.eps", bbox_inches="tight")

plt.show()

target_names = ["Not Raining", "Raining"]
print(classification_report(y_test, y_pred, target_names=target_names))

# %%
# SVM with rbf kernel
clf = svm.SVC(kernel="rbf")

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(
    "Test Accuracy:",
    metrics.accuracy_score(y_test, y_pred),
    "\nTrain Accuracy: ",
    metrics.accuracy_score(y_train, clf.predict(x_train)),
)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g", cbar=False)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.ylabel("Actual label", font1)
plt.xlabel("Predicted label", font1)


plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]

plt.savefig("svm_cm_undersampling_gauss_pca.eps", bbox_inches="tight")

plt.show()

target_names = ["Not Raining", "Raining"]
print(classification_report(y_test, y_pred, target_names=target_names))

# %%
# Split validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train.iloc[:1000]
partial_y_train = y_train.iloc[1000:]

epoch = 40

# Build network
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu",),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1, activation="sigmoid"),
    ]

)

# Set optimizer, loss function and metrics
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=epoch,
    batch_size=256,
    validation_data=(x_val, y_val),
    verbose=0
)

# Calculate accuracy
y_pred_proba = model.predict(x_test)

y_pred = list()

for i in y_pred_proba[:, 0]:
    if i > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Plot training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss_values, "b.", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("error_"+str(epoch)+".eps", bbox_inches="tight")
plt.show()

# # Retrain model with 20 epochs
# model = keras.Sequential(
#     [
#         layers.Dense(64, activation="relu",),
#         layers.Dropout(0.2),
#         layers.Dense(16, activation="relu"),
#         layers.Dropout(0.1),
#         layers.Dense(1, activation="sigmoid"),
#     ]
# )
# model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# history = model.fit(
#     partial_x_train,
#     partial_y_train,
#     epochs=20,
#     batch_size=256,
#     validation_data=(x_val, y_val),
# )

# y_pred_proba = model.predict(x_test)

# y_pred = list()

# for i in y_pred_proba[:, 0]:
#     if i > 0.5:
#         y_pred.append(1)
#     else:
#         y_pred.append(0)

# print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred))

# history_dict = history.history
# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, "bo", label="Training loss")
# plt.plot(epochs, val_loss_values, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig("error_20.eps", bbox_inches="tight")
# plt.show()

# %%
target_names = ["Not Raining", "Raining"]
print(classification_report(y_test, y_pred, target_names=target_names))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g", cbar=False)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.ylabel("Actual label", font1)
plt.xlabel("Predicted label", font1)


plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]

plt.savefig("NN_cm.eps", bbox_inches="tight")

plt.show()

# %%
# Plot the Neural Network Structure
plot_model(model, to_file="Network_structure.png", show_shapes=True)

# %%




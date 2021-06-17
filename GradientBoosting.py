import pandas as pd

# getting data
events = pd.read_csv("events.csv")

# dropping events that are not attempts
final_data = events[["id_event", "player", "location", "bodypart", "assist_method", "shot_outcome", "is_goal", "situation"]]
final_data = final_data.dropna(subset = ["shot_outcome"])

# Checking out frequency of usage of right foot, left foot and head
r_foot = (final_data.bodypart == 1)
right_foot = final_data.id_event[r_foot].count()
l_foot = (final_data.bodypart == 2)
left_foot = final_data.id_event[l_foot].count()
head = (final_data.bodypart == 3)
headers = final_data.id_event[head].count()
print(right_foot), print(left_foot), print(headers)

rem = [1, 2, 4, 5, 19]

player = []
location = []
body_part = []
assist_method = []
is_goal = []
shot_outcome = []
situation = []

# Retaining only the useful locations
for i in range(228498):
    if final_data.iloc[i][2] not in rem:
        player.append(final_data.iloc[i][1])
        location.append(final_data.iloc[i][2])
        body_part.append(final_data.iloc[i][3])
        assist_method.append(final_data.iloc[i][4])
        is_goal.append(final_data.iloc[i][6])
        situation.append(final_data.iloc[i][7])
        shot_outcome.append(final_data.iloc[i][5])

    else:
        continue

print(len(player))

ids = []
for i in range(len(player)):
    ids.append(i)

ids = pd.DataFrame(ids)
player = pd.DataFrame(player)
location = pd.DataFrame(location)
body_part = pd.DataFrame(body_part)
assist_type = pd.DataFrame(assist_method)
is_goal = pd.DataFrame(is_goal)
situation = pd.DataFrame(situation)
shot_outcome = pd.DataFrame(shot_outcome)


final = pd.concat([ids, player, situation, location, assist_type, body_part, is_goal, shot_outcome], axis = 1, keys = ['ids', 'player', 'situation', 'location', 'assist_type', 'body_part', 'is_goal', 'shot_outcome'])

# creating final dataframe to be used for models
data_bn = final[["assist_type", "situation", "body_part", "location", "is_goal"]]

data_bn.to_csv('data_bn.csv', index = False)
data_bn = pd.read_csv("data_bn.csv")
data_bn.drop(0, axis = 0, inplace = True)

# splitting into input and output
GB_y = pd.DataFrame((data_bn["is_goal"]))
GB_x = pd.concat(
    [pd.DataFrame(data_bn["situation"]), pd.DataFrame(data_bn["body_part"]), pd.DataFrame(data_bn["location"]),
     pd.DataFrame(data_bn["assist_type"])], axis=1)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(GB_x, GB_y, test_size=0.1)

# Training Gradient Boosting classifier

from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

classifier = [GradientBoostingClassifier()]
for clf in classifier:
    clf.fit(train_x, train_y)
    name = clf.__class__.__name__

    print("=" * 30)
    print(name)

    print('****Results****')
    # validation accuracy and testing accuracy
    train_predictions = clf.predict(train_x)
    test_predictions = clf.predict(test_x)
    train_probs = clf.predict_proba(train_x)
    acc_val = accuracy_score(train_y, train_predictions)
    acc_test = accuracy_score(test_y, test_predictions)
    print("Accuracy: {:.4%}".format(acc_val))
    print("Accuracy: {:.4%}".format(acc_test))

print("=" * 30)

# printing confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(train_y, train_predictions)

# testing for various shots:

# xG for Kante header
print(clf.predict_proba([[1, 3, 12, 2]]))

# xG for Pulisic miss
print(clf.predict_proba([[1, 1, 12, 4]]))

# xG for Ronaldo free kick
print(clf.predict_proba([[4, 1, 18, 0]]))


# getting all shots from el clasico 2016 (RM 2 - 1 FCB)

for i in range(811264, 811375):
    if events.iloc[i][5] == 1:
        print(events.iloc[i][:])

# sum xG for Real Madrid shots in 2-1 win over Barca in 2016

print(clf.predict_proba([[1, 1, 3, 0]])+clf.predict_proba([[1, 1, 10, 2]])+clf.predict_proba([[1, 1, 11, 0]])
      +clf.predict_proba([[1, 2, 11, 0]])+clf.predict_proba([[1, 1, 15, 1]])+ clf.predict_proba([[1, 2, 15, 1]])
      +clf.predict_proba([[3, 1, 15, 2]])+clf.predict_proba([[4, 1, 15, 1]]) + clf.predict_proba([[1, 2, 3, 1]])
      + clf.predict_proba([[1, 1, 3, 2]])+clf.predict_proba([[3, 1, 3, 2]]) + clf.predict_proba([[1, 1, 3, 4]])
      + clf.predict_proba([[1, 2, 10, 1]])+clf.predict_proba([[1, 2, 9, 0]]) + clf.predict_proba([[1, 1, 9, 1]]))

#

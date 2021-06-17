import numpy as np
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



# bayesian parameter estimate

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

model = BayesianModel([('situation', 'location'), ('assist_type', 'body_part'), ('body_part', 'location'), ('location', 'is_goal'), ('body_part', 'is_goal'), ('assist_type', 'is_goal'), ('situation', 'is_goal')])

# fitting our model
model.fit(data_bn, estimator=BayesianEstimator, prior_type="BDeu") # default equivalent_sample_size=5
for cpd in model.get_cpds():
    print(cpd)

# inference using variable elimination
from pgmpy.inference import VariableElimination

Xg_inference = VariableElimination(model)

# xG for Ronaldo free kick.
q1 = Xg_inference.query(variables=['is_goal'], evidence={'location': 18, 'body_part': 1, 'situation': 4, 'assist_type': 0})
print(q1)

# Kante header
q2 = Xg_inference.query(variables=['is_goal'], evidence={'location': 12, 'body_part': 3, 'assist_type': 2, 'situation': 1})
print(q2)

# Pulisic miss
q3= Xg_inference.query(variables=['is_goal'], evidence={'location': 3, 'body_part': 2, 'assist_type': 4, 'situation': 1})
print(q3)

# getting all shots from el clasico 2016 (RM 2 - 1 FCB)

for i in range(811264, 811375):
    if events.iloc[i][5] == 1:
        print(events.iloc[i][:])

# sum xG for Real Madrid shots in 2-1 win over Barca in 2016

q1 = Xg_inference.query(variables=['is_goal'], evidence={'location': 10, 'body_part': 1, 'situation': 1, 'assist_type': 1})
q2 = Xg_inference.query(variables=['is_goal'], evidence={'location': 3, 'body_part': 1, 'situation': 1, 'assist_type': 0})
q3 = Xg_inference.query(variables=['is_goal'], evidence={'location': 11, 'body_part': 1, 'situation': 1, 'assist_type': 0})
q4 = Xg_inference.query(variables=['is_goal'], evidence={'location': 11, 'body_part': 2, 'situation': 1, 'assist_type': 1})
q5 = Xg_inference.query(variables=['is_goal'], evidence={'location': 15, 'body_part': 1, 'situation': 1, 'assist_type': 1})
q6 = Xg_inference.query(variables=['is_goal'], evidence={'location': 15, 'body_part': 2, 'situation': 1, 'assist_type': 1})
q7 = Xg_inference.query(variables=['is_goal'], evidence={'location': 15, 'body_part': 1, 'situation': 3, 'assist_type': 2})
q8 = Xg_inference.query(variables=['is_goal'], evidence={'location': 15, 'body_part': 1, 'situation': 4, 'assist_type': 1})
q9 = Xg_inference.query(variables=['is_goal'], evidence={'location': 3, 'body_part': 2, 'situation': 1, 'assist_type': 1})
q10 = Xg_inference.query(variables=['is_goal'], evidence={'location': 3, 'body_part': 1, 'situation':1, 'assist_type': 2})
q11 = Xg_inference.query(variables=['is_goal'], evidence={'location': 3, 'body_part': 1, 'situation': 3, 'assist_type': 2})
q12 = Xg_inference.query(variables=['is_goal'], evidence={'location': 3, 'body_part': 1, 'situation': 1, 'assist_type': 1})
q13 = Xg_inference.query(variables=['is_goal'], evidence={'location': 10, 'body_part': 2, 'situation': 1, 'assist_type': 1})
q14 = Xg_inference.query(variables=['is_goal'], evidence={'location': 9, 'body_part': 2, 'situation': 1, 'assist_type': 0})
q15 = Xg_inference.query(variables=['is_goal'], evidence={'location': 9, 'body_part': 1, 'situation': 1, 'assist_type': 1})

print(q1+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14+q15)


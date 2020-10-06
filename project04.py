# compare algorithms
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load dataset
names = ['season','date','league_id','league','team1','team2','spi1','spi2','prob1','prob2','probtie','proj_score1','proj_score2','importance1','importance2','score1','score2','xg1','xg2','nsxg1','nsxg2','adj_score1','adj_score2']
dataset_spi_matches = read_csv('./data/spi_matches.csv', names=names)

dataset_spi_matches  = dataset_spi_matches.dropna()

linear_model = LinearRegression()
polynomial_model2 = make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
polynomial_model3 = make_pipeline(PolynomialFeatures(degree=3),LinearRegression())
polynomial_model4 = make_pipeline(PolynomialFeatures(degree=4),LinearRegression())
polynomial_model5 = make_pipeline(PolynomialFeatures(degree=5),LinearRegression())


dataset_spi_matches = dataset_spi_matches.drop(['season','date','league_id','league','probtie'], axis=1)
print(dataset_spi_matches)
label_encoder = preprocessing.LabelEncoder()

pyplot.plot(label_encoder.fit_transform(dataset_spi_matches['team1']),dataset_spi_matches['score1'],'x')
pyplot.ylabel('score1')
pyplot.xlabel('team1')
pyplot.axis([0, 450, 0, 15])
pyplot.show()

label_encoder = preprocessing.LabelEncoder()
pyplot.plot(label_encoder.fit_transform(dataset_spi_matches['team2']),dataset_spi_matches['score2'],'x')
pyplot.ylabel('score2')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 15])
pyplot.show()

pyplot.plot(dataset_spi_matches['prob1'],dataset_spi_matches['score1'],'o')
pyplot.ylabel('score1')
pyplot.xlabel('prob1')
pyplot.axis([0, 1, 0, 15])
pyplot.show()

pyplot.plot(dataset_spi_matches['prob2'],dataset_spi_matches['score2'],'o')
pyplot.ylabel('score2')
pyplot.xlabel('prob2')
pyplot.axis([0, 1, 0, 15])
pyplot.show()

######## For home

linear_model = linear_model.fit(label_encoder.fit_transform(dataset_spi_matches['team1']).reshape(-1, 1),dataset_spi_matches['score1'])
polynomial_model2 = polynomial_model2.fit(label_encoder.fit_transform(dataset_spi_matches['team1']).reshape(-1, 1),dataset_spi_matches['score1'])
polynomial_model3 = polynomial_model3.fit(label_encoder.fit_transform(dataset_spi_matches['team1']).reshape(-1, 1),dataset_spi_matches['score1'])
polynomial_model4 = polynomial_model4.fit(label_encoder.fit_transform(dataset_spi_matches['team1']).reshape(-1, 1),dataset_spi_matches['score1'])
polynomial_model5 = polynomial_model5.fit(label_encoder.fit_transform(dataset_spi_matches['team1']).reshape(-1, 1),dataset_spi_matches['score1'])

all_teams = np.array(list(set(dataset_spi_matches['team1'])))
# print(all_teams)
#Find average rating for all movies
predictions11 = linear_model.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions12 = polynomial_model2.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions13 = polynomial_model3.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions14 = polynomial_model4.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions15 = polynomial_model5.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))

print(predictions11)
print(predictions12)
print(predictions13)
print(predictions14)
print(predictions15)

pyplot.plot(label_encoder.fit_transform(all_teams),predictions11,'x')
pyplot.ylabel('predict score1')
pyplot.xlabel('team1')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions12,'x')
pyplot.ylabel('predict score1')
pyplot.xlabel('team1')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions13,'x')
pyplot.ylabel('predict score1')
pyplot.xlabel('team1')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions14,'x')
pyplot.ylabel('predict score1')
pyplot.xlabel('team1')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions15,'x')
pyplot.ylabel('predict score1')
pyplot.xlabel('team1')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()


############### For away


linear_model = linear_model.fit(label_encoder.fit_transform(dataset_spi_matches['team2']).reshape(-1, 1),dataset_spi_matches['score2'])
polynomial_model2 = polynomial_model2.fit(label_encoder.fit_transform(dataset_spi_matches['team2']).reshape(-1, 1),dataset_spi_matches['score2'])
polynomial_model3 = polynomial_model3.fit(label_encoder.fit_transform(dataset_spi_matches['team2']).reshape(-1, 1),dataset_spi_matches['score2'])
polynomial_model4 = polynomial_model4.fit(label_encoder.fit_transform(dataset_spi_matches['team2']).reshape(-1, 1),dataset_spi_matches['score2'])
polynomial_model5 = polynomial_model5.fit(label_encoder.fit_transform(dataset_spi_matches['team2']).reshape(-1, 1),dataset_spi_matches['score2'])

all_teams = np.array(list(set(dataset_spi_matches['team2'])))
# print(all_teams)
#Find average rating for all movies
predictions21 = linear_model.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions22 = polynomial_model2.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions23 = polynomial_model3.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions24 = polynomial_model4.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))
predictions25 = polynomial_model5.predict(label_encoder.fit_transform(all_teams).reshape(-1, 1))

print(predictions21)
print(predictions22)
print(predictions23)
print(predictions24)
print(predictions25)

pyplot.plot(label_encoder.fit_transform(all_teams),predictions21,'x')
pyplot.ylabel('predict score2')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions22,'x')
pyplot.ylabel('predict score2')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions23,'x')
pyplot.ylabel('predict score2')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions24,'x')
pyplot.ylabel('predict score2')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),predictions25,'x')
pyplot.ylabel('predict score2')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()


#### combine two weights
p1 = (predictions11 + predictions21)/2
p2 = (predictions12 + predictions22)/2
p3 = (predictions13 + predictions23)/2
p4 = (predictions14 + predictions24)/2
p5 = (predictions15 + predictions25)/2

pyplot.plot(label_encoder.fit_transform(all_teams),p1,'x')
pyplot.ylabel('predict score')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),p2,'x')
pyplot.ylabel('predict score')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),p3,'x')
pyplot.ylabel('predict score')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),p4,'x')
pyplot.ylabel('predict score')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

pyplot.plot(label_encoder.fit_transform(all_teams),p5,'x')
pyplot.ylabel('predict score')
pyplot.xlabel('team2')
pyplot.axis([0, 450, 0, 1.8])
pyplot.show()

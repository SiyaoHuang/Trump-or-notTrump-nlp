import IO
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

xTr, _,yTr = IO.get_rd('train.csv', 5000)
print('start')
clf = RandomForestClassifier(bootstrap=True, n_estimators = 500,max_features=None, max_depth =3,random_state = 0)
scores = cross_val_score(clf, xTr, yTr, cv=5)
print(scores.mean())
clf.fit(xTr,yTr)
test,_ = IO.get_rdt2('test.csv', 5000)
predicted = clf.predict(test)
print(predicted)
for i in predicted:
    if i == 0:
        print(-1)
    else:
        print(1)
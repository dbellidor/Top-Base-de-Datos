#
#  FILTERINGDATA.py
#
#  Code file for the book Programmer's Guide to Data Mining
#  http://guidetodatamining.com
#  Ron Zacharski
#

from math import sqrt

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5, "Phoenix": 5.0, "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0, "Vampire Weekend": 1.0},
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0, "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5, "The Strokes": 3.0}
        }


def cos_similar(rating1, rating2):
    vec_length_1 = 0
    vec_length_2 = 0
    dot_product = 0

    include_all_occurrences(rating1, rating2)

    # print("rating1 ==> ", rating1)

    x,y = create_vectors(rating1, rating2)

    # print("x ==> ", x)
    # print("y ==> ", y)

    for rate in x:
        vec_length_1 += rate ** 2

    for rate in y:
        vec_length_2 += rate ** 2

    for rate1, rate2 in zip(x,y):
        dot_product += rate1 * rate2

    if vec_length_1 == 0 or vec_length_2 == 0:
        return 0

    # vec_length_1 = sqrt(vec_length_1)
    print("Longitud vector 1 antes de sqrt: ", vec_length_1)
    vec_length_1 = sqrt(vec_length_1)
    print("Longitud vector 2 antes de sqrt: ", vec_length_2)
    vec_length_2 = sqrt(vec_length_2)
    similarity = dot_product / (vec_length_1 * vec_length_2)
    print("Producto punto: ", dot_product)
    print("Longitud vector 1: ", vec_length_1)
    print("Longitud vector 2: ", vec_length_2)
    print("Similaridad: ", similarity)

    return similarity


def include_all_occurrences(rating1, rating2):
    vector_x = []
    vector_y = []

    for key in rating1:
        if key not in rating2:
            rating2.setdefault(key, 0)

    for key in rating2:
        if key not in rating1:
            rating1.setdefault(key, 0)

    # l1 = sorted(list(rating1.items()))
    # l2 = sorted(list(rating2.items()))

    # print(l1)
    # print(l2)

    return


def create_vectors(rating1, rating2):
    vector_x = []
    vector_y = []

    l1 = sorted(list(rating1.items()))
    l2 = sorted(list(rating2.items()))

    for (a,b) in l1:
        vector_x.append(b)

    for (a,b) in l2:
        vector_y.append(b)

    # print("vector_x ==> ", vector_x)
    # print("vector_y ==> ", vector_y)

    return vector_x, vector_y


def manhattan(rating1, rating2):
    """Computes the Manhattan distance. Both rating1 and rating2 are dictionaries
       of the form {'The Strokes': 3.0, 'Slightly Stoopid': 2.5}"""
    distance = 0
    total = 0
    for key in rating1:
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])
            total += 1
    if total > 0:
        return distance / total
    else:
        return -1 #Indicates no ratings in common

def minkowski(rating1, rating2, r):
    """Computes the Minkowski distance.  
    Both rating1 and rating2 are dictionaries of the form
    {'The Strokes': 3.0, 'Slightly Stoopid': 2.5}"""
    distance = 0
    commonRatings = False
    for key in rating1:
        if key in rating2:
            distance +=pow(abs(rating1[key] - rating2[key]), r)
	    commonRatings = True
    if commonRatings:
        return pow(distance, 1/r)
    else:
        return 0 #Indicates no ratings in common

def pearson(rating1, rating2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
    # now compute denominator
    denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) / n) / denominator
            

def computeNearestNeighbor(username, users):
    """creates a sorted list of users based on their distance to username"""
    distances = []
    for user in users:
        if user != username:
            distance = manhattan(users[user], users[username])
            distances.append((distance, user))
    # sort based on distance -- closest first
    distances.sort()
    return distances

def recommend(username, users):
    """Give list of recommendations"""
    # first find nearest neighbor
    nearest = computeNearestNeighbor(username, users)[0][1]

    recommendations = []
    # now find bands neighbor rated that user didn't
    neighborRatings = users[nearest]
    userRatings = users[username]
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendations.append((artist, neighborRatings[artist]))
    # using the fn sorted for variety - sort is more efficient
    return sorted(recommendations, key=lambda artistTuple: artistTuple[1], reverse = True)

print ("Coseno")
print(cos_similar(users['Angelica'], users['Veronica']))
print ("Manhattan")
print(manhattan(users['Angelica'], users['Veronica']))

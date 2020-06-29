import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# first five motivational quotes, next 5 benifits of eating fruits, next 5 views on dogs
text = ["Each time we face our fear, we gain strength,courage, and confidence in the doing.",
        "If you hear a voice within you say, you cannot paint, then by all means paint, and what voice will be silenced",
        "Don't be satisfied with stories, how things have gone with others, unfold your own myth",
        "you wouldn't worry so much about what others think of you if you realized how seldom they do",
        "It is confidence in our bodies, minds, and spirits that allows us to keep looking for new adventures",
        "Eating a diet rich in vegetables and fruits as part of an overall healthy diet may reduce risk of heart disease.",
        "A fruit cointaining eating pattern is part of overall healthy diet and may protect aganist certain cancers.",
        "Eating a diet rich in fruit may reduce risk for stroke, other cardiovascular diseases and type-2 diabetes.",
        "Vitamin C in clementine prevents premature aging by fighting against free radicals and promoting collagen production",
        "Two phytonutrients in honeydew melon, lutein, and zeaxanthin, are essential for maintaining eye health as we get older.",
        " dogs one of the cutest animals on earth (even the bitey ones) and one of the most beautiful.",
        "Dogs don’t just fill your heart; they actually make it stronger.",
        "Research has repeatedly found that daily dog walks help you lose weight, since they force you to into moderate physical activity for 10, 20, and even 30 minutes at a time.",
        "There’s a reason therapy dogs are so effective",
        "If you’re over 65 and own a pet, odds are you seek medical help about 30 percent less often than people who don’t have a pet",
        ]

vectorizer = CountVectorizer()

# tokenize and build the vocabulary
vectorizer.fit(text)


# encode the document
vector = vectorizer.transform(text)

countVector = vector.toarray()

def kmeans(x):
    n = countVector.shape[1]
    # initialize centroids randomly
    centroid_1 = np.random.rand(1,n)
    centroid_2 = np.random.rand(1, n)
    centroid_3 = np.random.rand(1, n)

    labels = None

    i = 0

    while i<100:
        #compute the distance of each pt from the centroids
        distance_1 = np.sum(np.square(x-centroid_1), keepdims=True, axis=1)
        distance_2 = np.sum(np.square(x - centroid_2), keepdims=True, axis=1)
        distance_3 = np.sum(np.square(x - centroid_3), keepdims=True, axis=1)

        labels = np.argmin(np.hstack([distance_1, distance_2, distance_3]), axis=1)

        class_1 = x[labels==0, :]
        class_2 = x[labels==1, :]
        class_3 = x[labels==2, :]

        #update centroids
        if class_1.shape[0]>0:
            centroid_1 = np.mean(class_1, axis=0)

        if class_2.shape[0]>0:
            centroid_2 = np.mean(class_2, axis=0)

        if class_3.shape[0]>0:
            centroid_3 = np.mean(class_3, axis=0)

        i += 1

    return labels


print(kmeans(countVector))







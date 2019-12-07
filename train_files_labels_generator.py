import os
import pandas as pd

train_images = 'train/'

files = os.listdir(train_images)
files = [os.path.join(train_images, x) for x in files]
files.sort()
cat_or_dog = map(lambda file: 0 if "cat" in file else 1, files)

df = pd.DataFrame(cat_or_dog)
df.to_csv('train_labels.csv', index_label = 'id', header = ['label'])
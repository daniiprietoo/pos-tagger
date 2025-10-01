from conllu import parse_incr
from utils import download_datasets

# Download datasets
dev, train, test = download_datasets()

# load data
data_dev = list(parse_incr(open(dev, "r", encoding="utf-8")))
data_train = list(parse_incr(open(train, "r", encoding="utf-8")))
data_test = list(parse_incr(open(test, "r", encoding="utf-8")))

# select columns 1 and 4 to keep word and pos tag
# Ignore multi-word and empty token if decimal -> greater than 0
def select_columns(data):
    selected_data = []
    for sentence in data:
        selected_sentence = []
        for token in sentence:
            if isinstance(token["id"], int) and token["form"].strip() != "":
                selected_sentence.append((token["form"], token["upostag"]))
        selected_data.append(selected_sentence)
    return selected_data

train_data = select_columns(data_train)
dev_data = select_columns(data_dev)
test_data = select_columns(data_test)

print(dev_data[0:2])
print(train_data[0:2])

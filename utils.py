from keras.utils import get_file
from conllu import parse_incr

# Download datasets from URLs and save them locally
def download_datasets_english():
    url1 = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/refs/heads/master/en_ewt-ud-dev.conllu"
    path1 = get_file(
        origin=url1,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False, # True -> re-download the file
    )

    url2 = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/refs/heads/master/en_ewt-ud-train.conllu"
    path2 = get_file(
        origin=url2,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False,
    )

    url3 = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/refs/heads/master/en_ewt-ud-test.conllu"
    path3 = get_file(
        origin=url3,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False,
    )
    return path1, path2, path3

def download_datasets_german():
    url1 = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-dev.conllu"
    path1 = get_file(
        origin=url1,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False, # True -> re-download the file
    )

    url2 = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-train.conllu"
    path2 = get_file(
        origin=url2,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False,
    )

    url3 = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-test.conllu"
    path3 = get_file(
        origin=url3,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False,
    )
    return path1, path2, path3

def download_datasets_spanish():
    url1 = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/refs/heads/master/es_gsd-ud-dev.conllu"
    path1 = get_file(
        origin=url1,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False, # True -> re-download the file
    )

    url2 = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/refs/heads/master/es_gsd-ud-train.conllu"
    path2 = get_file(
        origin=url2,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False,
    )

    url3 = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/refs/heads/master/es_gsd-ud-test.conllu"
    path3 = get_file(
        origin=url3,
        cache_dir="./",  # Saves it to ./datasets/name_of_the_file
        force_download=False,
    )
    return path1, path2, path3

def load_data(language: str):
    dev_path, train_path, test_path = '', '', ''
    if language == "spanish":
        dev_path, train_path, test_path = download_datasets_spanish()
    elif language == "german":
        dev_path, train_path, test_path = download_datasets_german()
    else:
        dev_path, train_path, test_path = download_datasets_english()

    dev_data = list(parse_incr(open(dev_path, "r", encoding="utf-8")))
    train_data = list(parse_incr(open(train_path, "r", encoding="utf-8")))
    test_data = list(parse_incr(open(test_path, "r", encoding="utf-8")))

    return train_data, dev_data, test_data

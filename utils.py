from keras.utils import get_file

# Download datasets from URLs and save them locally
def download_datasets():
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

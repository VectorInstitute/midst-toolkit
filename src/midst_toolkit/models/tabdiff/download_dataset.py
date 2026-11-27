import os
import zipfile
from urllib import request


DATA_DIR = "data"


NAME_URL_DICT_UCI = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "news_nocat": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    # Renamed away from "diabetes" so UCI/CDTD clinical does not collide with MIDST fixtures.
    "diabetes_cdtd_legacy": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
}


def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):
    print(f"Start processing dataset {name} from UCI.")
    save_dir = f"{DATA_DIR}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f"{save_dir}/{name}.zip")
        print(f"Finish downloading dataset from {url}, data has been saved to {save_dir}.")

        unzip_file(f"{save_dir}/{name}.zip", save_dir)
        print(f"Finish unzipping {name}.")

    else:
        print("Aready downloaded.")


if __name__ == "__main__":
    for name in NAME_URL_DICT_UCI:
        download_from_uci(name)

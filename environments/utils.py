import os

def create_if_not_exists(folder_name):
    print("Creating folder '{}'...".format(folder_name))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
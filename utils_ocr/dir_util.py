#
#  Copyright by Hyperlogy Corporation, 2020
#  Smart eKYC project.
#
import os
import config_ocr.constants as const


def dir_explore(input_dir, explore_mode):
    """ This script will explore a directory then return a list of image files within it.
        The script does not return files within its sub-folder(s).
    Args:
        input_dir: the input directory
        explore_mode: 'convert', 'filter' or 'predict'
        - 'convert' (using in pre-processing): convert pdf to images format.
        - 'filter' (using in pre-processing): filter gray / binary images.
        - 'browse' (using when train/predict data): get list of image files.
    Returns: list of files, including filename and corresponding extension
        - with 'convert' mode: return all PDF files and paths in root and child folders.
        - with 'filter' mode: return all IMAGE files and paths in root and child folders.
        - with 'browse' mode: just return IMAGE files in root folder.
    """
    # Co hai cach liet ke danh sach file trong mot thu muc: os.listdir() / os.walk()
    # nhung cach thu nhat khong phan biet file va folder, nen ta dung cach thu hai.
    explore_list = []
    for directory, subdirs, files in os.walk(input_dir):
        # print("\ndirectory = {} \n subdirs = {} \n files = {}\n".format(directory, subdirs, files))
        if explore_mode in [const.EXPLORE_CONVERT, const.EXPLORE_FILTER]:
            for file in files:
                explore_list.append([file, directory])
        elif explore_mode == const.EXPLORE_BROWSE and directory == input_dir:
            explore_list = files
            break
    # print("HYPERLOGY dir_explore({})\nexplore_list = {}".format(input_dir, explore_list))

    # bat dau lay ra danh sach file, tuy theo tung che do
    files_list = []
    for explore in explore_list:

        if explore_mode == const.EXPLORE_CONVERT:  # convert file PDF
            file = explore[0]
            if file[-3:] == 'pdf':
                files_list.append([file[:-4], file[-4:], explore[1]])  # luu ten file, phan mo rong va path

        elif explore_mode == const.EXPLORE_FILTER:  # loc file anh
            file = explore[0]
            # phan mo rong la 'tiff'
            if file[-4:] in const.IMAGE_EXTENSION_LIST:
                files_list.append([file[:-5], file[-5:], explore[1]])  # luu ten file, phan mo rong va path
            # phan mo rong la 'bmp', 'gif', 'jpg', 'png'
            elif file[-3:] in const.IMAGE_EXTENSION_LIST:
                files_list.append([file[:-4], file[-4:], explore[1]])  # luu ten file, phan mo rong va path

        elif explore_mode == const.EXPLORE_BROWSE:  # lay ra file anh de train/predict
            # phan mo rong la 'tiff'
            if explore[-4:] in const.IMAGE_EXTENSION_LIST:
                files_list.append([explore[:-5], explore[-5:]])  # luu ten file va phan mo rong
            # phan mo rong la 'bmp', 'gif', 'jpg', 'png'
            elif explore[-3:] in const.IMAGE_EXTENSION_LIST:
                files_list.append([explore[:-4], explore[-4:]])  # luu ten file va phan mo rong

    # print("\nHYPERLOGY dir_explore({})\nfiles_list = {}\nTotal : {} files".format(input_dir, files_list, len(files_list)))
    return files_list


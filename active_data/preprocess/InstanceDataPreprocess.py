import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread
from commons.utils import *
from active_data.constant import *


def make_dataset(mode, root, data_name=GLAND_DATA, radio=0.89, partA='A'):
    if mode == "test":
        if data_name == GLAND_DATA:
            root = root
        if data_name == MONUSEG_DATA:
            root = join(root, 'MoNuSegTestData')
        if data_name == CRAG_DATA:
            root = join(root, 'valid')
        if data_name == TNBC_DATA:
            root = join(root, 'test')
        img_list = get_test_image_list(original_data_dir=root, data_name=data_name, partA=partA)
        img_list = sorted(img_list)
        # print(img_list)
    else:
        if data_name == GLAND_DATA:
            root = root
        if data_name == MONUSEG_DATA:
            root = join(root, 'MoNuSeg Training Data')
        if data_name == CRAG_DATA:
            root = join(root, 'train')
        if data_name == TNBC_DATA:
            root = join(root, 'train')
        img_list = get_full_image_list(original_data_dir=root, data_name=data_name, is_full_image=True)
        img_list = sorted(img_list)
        if data_name == MONUSEG_DATA:
            radio = 0.9  #old 2018 images with test dat
            # radio = 0.8 #old 2017 images with test da
        if data_name == CRAG_DATA:
            radio = 0.89
        if data_name == GLAND_DATA:
            radio = 0.89
        if data_name == TNBC_DATA:
            radio = 0.82
        n = len(img_list)
        if mode == "train":
            img_list = img_list[0:int(n * radio)]
        elif mode == "valid":
            img_list = img_list[int(n * radio):]
            # print(img_list)
        else:
            raise ValueError('Dataset split specified does not exist!')
    items = []
    print('load %s data %d image' % (mode, len(img_list)))
    for (im_p, im_t) in img_list:
        item = (im_p, im_t, im_p.split('/')[-1])
        items.append(item)
    return items

def split_train_val_test(orig_path='../../../medical_data/Gland/Warwick_QU/', val_size=0.1):
    """Split image names into training set and validation set.
    """
    grade = pd.read_csv(join(orig_path, 'Grade.csv'))
    grade.drop(grade.columns[1:3], axis=1, inplace=True)

    # testA_set = grade[grade['name'].str.startswith('testA_')]['name']
    # testB_set = grade[grade['name'].str.startswith('testB_')]['name']

    grade = grade[grade['name'].str.startswith('train_')]
    grade.columns = ('name', 'grade')
    grade['grade'] = pd.factorize(grade['grade'])[0]

    x, y = grade['name'], grade['grade']
    # train_set, val_set, _, _ = train_test_split(
    #     x, y, test_size=val_size, stratify=y)

    return train_test_split(
        x, y, test_size=val_size, stratify=y)


def get_test_image_list(original_data_dir, data_name=GLAND_DATA, partA='A'):
    post_fix = '.png'
    if data_name == GLAND_DATA:
        post_fix = 'anno.bmp'
        if partA == 'A':
            img_dir = '{:s}/{:s}'.format(original_data_dir, 'testimgA')
            target_dir = '{:s}/{:s}'.format(original_data_dir, 'testlabelsA')
        elif partA == 'B':
            img_dir = '{:s}/{:s}'.format(original_data_dir, 'testimgB')
            target_dir = '{:s}/{:s}'.format(original_data_dir, 'testlabelsB')
    if data_name == MONUSEG_DATA:
        if partA == 'A':
            img_dir = '{:s}/{:s}'.format(original_data_dir, 'diff/images')
            target_dir = '{:s}/{:s}'.format(original_data_dir, 'diff/masks')
        elif partA == 'B':
            img_dir = '{:s}/{:s}'.format(original_data_dir, 'same/images')
            target_dir = '{:s}/{:s}'.format(original_data_dir, 'same/masks')
    if data_name == CRAG_DATA:
        img_dir = '{:s}/{:s}'.format(original_data_dir, 'Images')
        target_dir = '{:s}/{:s}'.format(original_data_dir, 'Annotation')
    if data_name == TNBC_DATA:
        img_dir = '{:s}/{:s}'.format(original_data_dir, 'imgs')
        target_dir = '{:s}/{:s}'.format(original_data_dir, 'gts')
    dir_list = [img_dir, target_dir]
    img_list = []
    if len(dir_list) == 0:
        return img_list

    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img), ]
        for i in range(1, len(img_filename_list)):
            if data_name == GLAND_DATA:
                img_name = '{:s}_{:s}'.format(img1_name, post_fix)
            else:
                img_name = '{:s}{:s}'.format(img1_name, post_fix)
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


def get_full_image_list(original_data_dir, data_name=GLAND_DATA, is_full_image=False):
    if data_name == GLAND_DATA:
        img_dir = '{:s}/{:s}'.format(original_data_dir, 'oriimgs')
        target_dir = '{:s}/{:s}'.format(original_data_dir, 'labels')
    if data_name == MONUSEG_DATA:
        img_dir = '{:s}/{:s}'.format(original_data_dir, 'Tissue Images')
        target_dir = '{:s}/{:s}'.format(original_data_dir, 'labels')
    if data_name == CRAG_DATA:
        img_dir = '{:s}/{:s}'.format(original_data_dir, 'Images')
        target_dir = '{:s}/{:s}'.format(original_data_dir, 'labels')
    if data_name == TNBC_DATA:
        img_dir = '{:s}/{:s}'.format(original_data_dir, 'imgs')
        target_dir = '{:s}/{:s}'.format(original_data_dir, 'labels')
    dir_list = [img_dir, target_dir]
    img_list = []
    if len(dir_list) == 0:
        return img_list

    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img), ]
        for i in range(1, len(img_filename_list)):
            img_name = '{:s}_{:s}'.format(img1_name, 'label.png')
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


def make_test_dataset(dir='../../../medical_data/Gland/Warwick_QU',
                      data_name=GLAND_DATA, partA='A'):
    training_list = get_test_image_list(original_data_dir=dir, data_name=data_name, partA=partA)
    dataset = training_list
    image_name = [basename(x[0]) for x in dataset]
    image_paths = [x[0] for x in dataset]
    image_gt = [x[1] for x in dataset]
    image_label = [1 for x in dataset]
    return image_paths, image_gt, image_label, image_name


def make_train_dataset(dir='../../../medical_data/Gland/Warwick_QU/patches',
                       post_fix=['label.png'], data_name=GLAND_DATA, is_full_image=False, seed=666):
    if not is_full_image:
        if data_name == GLAND_DATA:
            dir = join(dir, 'patches')
        if data_name == MONUSEG_DATA:
            dir = join(dir, 'MoNuSeg Training Data/patches')
        if data_name == CRAG_DATA:
            dir = join(dir, 'train/patches')
        if data_name == TNBC_DATA:
            dir = join(dir, 'train/patches')
        training_list = get_imgs_list(original_data_dir=dir, post_fix=post_fix, seed=seed)
        dataset = training_list
        image_name = [basename(x[0]) for x in dataset]
        image_paths = [x[0] for x in dataset]
        image_gt = [x[1] for x in dataset]
        image_label = [1 for x in dataset]
        return image_paths, image_gt, image_label, image_name
    else:
        dataset = make_dataset('train', dir, data_name=data_name)
        image_name = [im_n for (im_p, im_t, im_n) in dataset]
        image_paths = [im_p for (im_p, im_t, im_n) in dataset]
        image_gt = [im_t for (im_p, im_t, im_n) in dataset]
        image_label = [1 for (im_p, im_t, im_n) in dataset]
        return image_paths, image_gt, image_label, image_name


def make_vali_dataset(dir='../../../medical_data/Gland/Warwick_QU', data_set=GLAND_DATA):
    dataset = make_dataset('valid', dir, data_name=data_set)
    image_name = [im_n for (im_p, im_t, im_n) in dataset]
    image_paths = [im_p for (im_p, im_t, im_n) in dataset]
    image_gt = [im_t for (im_p, im_t, im_n) in dataset]
    image_label = [1 for (im_p, im_t, im_n) in dataset]
    return image_paths, image_gt, image_label, image_name


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 'tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# get the image list pairs
def get_imgs_list(original_data_dir='../../../medical_data/Gland/Warwick_QU/patches', post_fix=['anno_label.png'],
                  seed=666):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :param post_fix: e.g. ['label.png', 'weight.png',...]
    :return: e.g. [(img1.ext, img1_label.png, img1_weight.png), ...]
    """
    img_dir = '{:s}/{:s}'.format(original_data_dir, 'images')
    target_dir = '{:s}/{:s}'.format(original_data_dir, 'labels')
    dir_list = [img_dir, target_dir]
    img_list = []
    if len(dir_list) == 0:
        return img_list
    if len(dir_list) != len(post_fix) + 1:
        raise (RuntimeError('Should specify the postfix of each img type except the first input.'))

    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img), ]
        for i in range(1, len(img_filename_list)):
            img_name = '{:s}_{:s}'.format(img1_name, post_fix[i - 1])
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))
    np.random.seed(seed)
    np.random.shuffle(img_list)
    return img_list

import os
import numpy as np

##########################
# return train, test list for AID Dataset
##########################
def split_trainval_test(input_dir, dic):
    all_file = []
    train = []
    test = []
    current_index = 1
    num_classes = 30 #determine by corresponding dataset

    for cls_name in os.listdir(input_dir):
        if cls_name == ".DS_Store":
            continue
        file_dir = os.path.join(input_dir, cls_name)
        for img_name in os.listdir(file_dir):
            if img_name.endswith(".jpg"):
                src_img = os.path.normpath(file_dir.split('/')[-1])
                img_path = os.path.join(src_img, img_name)
                all_file.append(img_path)
    
    all_file.sort()
    tmp = np.zeros((num_classes,), dtype = np.int)
    label_current_index = 0

    for data in all_file:
        img_label = os.path.normpath(data.split('/')[0])
        if img_label in dic.keys():
            tmp[dic[img_label]] += 1
            continue;
        else:
            dic[img_label] = label_current_index
            label_current_index += 1
            tmp[dic[img_label]] += 1


    for data in all_file:
        is_test_item = False
        img_label = os.path.normpath(data.split('/')[0])
        # train ratio depend on previous works
        train_size = int(0.5 * tmp[dic[img_label]])
        if current_index > train_size:
            is_test_item = True

        item = data
        if is_test_item:
            test.append(item)
        else:
            train.append(item)

        current_index += 1
        if current_index > tmp[dic[img_label]]:
            current_index = 1

    return train, test, dic

def generate_train_val_test_txt_file(train, test, save_dir, dic):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_txt_path = os.path.join(save_dir, "train.txt")
    test_txt_path  = os.path.join(save_dir, "test.txt")

    train_str = ""
    test_str  = ""

    for img_path in train:
        label = dic[os.path.normpath(img_path.split('/')[0])]
        train_str += img_path + ' ' + str(label) + "\n"
    for img_path in test:
        label = dic[os.path.normpath(img_path.split('/')[0])]
        test_str += img_path + ' ' + str(label) + "\n"

    with open(train_txt_path, "w") as fw:
        fw.write(train_str)
    with open(test_txt_path, "w") as fw:
        fw.write(test_str)


if __name__ == "__main__":
    dic = {}
    isCaseW = False
    if isCaseW:
        input_dir = "/content/drive/My Drive/data/AID"
        save_dir  = "/content/drive/My Drive/data/"
    else:
        input_dir = "/content/drive/My Drive/data/AID"
        save_dir  = "/content/drive/My Drive/data/"
        train, test, dic_new = split_trainval_test(input_dir, dic)
        generate_train_val_test_txt_file(train, test, save_dir, dic_new)
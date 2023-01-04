import cv2 as cv
import os

def sort_img_dataset():
    ans_path = r'captcha_ans/captcha_ans.txt'
    input_path = r'captcha/'
    output_path = r'img_data/'
    all_files = []
    all_ans = []
    count = 0
    total = 0

    # get all images from captcha
    for p, d, f in os.walk(input_path):
        all_files = f
        total = len(f)
        break

    # read captcha answers
    with open(ans_path, 'r', encoding='utf-8') as r:
        all_ans = [e.strip() for e in r.read().strip().split('\n')]

    print('Sort files to captcha_imgs folder...')
    print('Total: {}'.format(total))
    
    # save images into output path, use answers to name them
    for img_name in all_files:
        img_to_ans_num = int(img_name.split('.')[0][7:])
        cv.imwrite('{}{}.png'.format(output_path, all_ans[img_to_ans_num]), cv.imread(input_path + img_name, flags=cv.IMREAD_UNCHANGED))
        count += 1
        if count % 1000 == 0:
            print('Now {}/{}'.format(count, total))
    
if __name__ == '__main__':
    sort_img_dataset()
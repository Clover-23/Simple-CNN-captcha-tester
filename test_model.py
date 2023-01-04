import tensorflow as tf
import numpy as np
import cv2

# testing model example
if __name__ == '__main__':
    m = tf.keras.models.load_model('captcha_model.h5')
    img_gray = np.array([cv2.cvtColor(cv2.imread('captcha/{}'.format(input('File (in "captcha" folder): '))), cv2.COLOR_BGR2GRAY).reshape(20, 60, 1) / 255.0])
    
    # prediction
    ans = m.predict(img_gray)
    
    # use same label code
    label_code = '9876543210ZYXWVUTSRQPONMLKJIHGFEDCBA'
    for ele in ans:
        mx = 0
        index = 0
        for i in range(36):
            if ele[0][i] > mx:
                mx = ele[0][i]
                index = i
        print(label_code[index], end='')
    print()
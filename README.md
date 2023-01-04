# 簡易CNN實作 - 驗證碼圖像辨識

透過難易度不高的且好取得的驗證碼圖片作為對象，做卷積神經網路（CNN）的實作練習。

## 實作過程

透過簡單的爬蟲取得驗證碼圖片，蒐集至六位數的量（含程式碼，但不含取得認證碼圖片的網址）。
將圖片的檔案名稱改為其驗證碼的答案，以方便實作。
資料（圖片）進入模型之前，轉成灰階後標準化，也就是簡易的資料前處理。
由於本次實作的圖片解析度不大，加上任務難度不高，所以訓練迅速，主要難點還是在單一input必須多個output，解決之後就沒有問題了。

## 概述

本次實作包刮任務以及模型架構本身都相對簡易，也沒有使用到高深的資料前處理，
不過，在複雜的圖像辨識上，若能修改輸入輸出，微調程式碼，並增加本實作的架構深度，並拉長訓練時間，在其他難度較高任務上應會有不錯的表現，
基本上如同個小模板和範例一樣，供參考、學習、以及延伸。

## 檔案說明
由於檔案數量與大小太大緣故，只附上部分訓練資料。
* **Simple_Captcha_CNN**
  * **captcha**
    * origin images example
  * **captcha_ans**
    * captcha_ans.txt
  * **img_data**
      * captcha_img_data.zip(檔案太大未附)
      * renamed images example
  * captcha_model.h5
  * cnn_captcha_test.py
  * png_worm.py
  * sort_captcha_img.py
  * test_model.py

**cnn_captcha_test.py** 為訓練model使用的程式碼
**png_worm.py** 為取得驗證碼圖片的爬蟲程式碼
**sort_captcha_img.py** 其功能為將captcha資料夾中的圖片對應到captcha_ans.txt中的答案，將答案當成檔名重新存檔至img_data資料夾
**test_model.py**測試model，也是該模型的使用範例

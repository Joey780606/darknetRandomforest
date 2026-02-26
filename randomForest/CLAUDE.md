# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

作Random forest的分類器. 使用YOLOv3/Darknet訓練完的weight, 5 facial landmarks (left_eye, right_eye, nose_tip, left_mouth, right_mouth), 開啟電腦的webcam,若偵測到人臉,用Randome forest的分類器判斷是誰,若判斷不到,讓使用者輸入名字,從webcam定期抓frame做分類器訓練.

## Architecture
1. YOLOv3/Darknet 參考資料: https://github.com/AlexeyAB/darknet
2. Random forest來源:由你決定

## Code Specification
1. 開發語言: Python 3.13
2. UI library: customtkinter
3. 註解請用中文.
4. Variable Naming: CamelCase is used consistently
5. Error Handling: All API calls must include a try-catch block.

## Design decisions / Constraints
1. 程式開啟時,若Webcam沒有開啟,就把Webcam打開. 程式關閉前,先把Webcam關掉.

## UI Design
規劃UI有四rows,
第一row,從左到右有以下元件:
1.Button,名稱叫btnDetectName. 內容為:Detect face. 若被按下,內容改成Stop detect face,並開始定時截取Webcam的frame,傳到Random Forest進行分類檢查.若能找到分類的名字,就顯示在下方DetectName Label裡.偵測時間10秒,確定找不到就跳dialog顯示找不到.結束後,內容再改成Detect face.
2.Label,名稱叫lblDetectName. 用來顯示 btnDetectName button後,若有找到分類的名稱,就顯示出來.若找不到,就可以填寫 "Not found".
第二row,從左到右有以下元件:
1.Textblock,名稱叫tblMyName. 供操作者輸入姓名.
2.Button,名稱叫btnLearn,內容寫:Learning. 若被按下,開始定時截取Webcam的frame,使用YOLOv3/Darknet訓練完的weights偵測到人臉(只要5個facial landmarks偵測到3個以上),將frame傳到Random Forest進行分類學習的工作,學習時間為1分鐘.
第三row,是放Webcam可以看到的畫面,若能偵測到像眼睛,嘴巴等,都要能在畫面上,標示出來讓使用者看到.
第四row,放Label,名稱叫lblRemain. 內容為: Remaining study seconds: XX.  XX就要顯示剩餘的顯示秒數.

## Exist file and function
1. cfg\face_landmark.names : Facial landmarks的名稱.
2. cfg\yolov3-face-landmark.cfg : YOLOv3/Darknet的組態.
3. weights\yolov3-face-landmark_final.weights : YOLOv3/Darknet訓練完的weight.
4. inference.py : 使用Darknet訓練權重來抓取webcam frame的facial landmarks的程式.


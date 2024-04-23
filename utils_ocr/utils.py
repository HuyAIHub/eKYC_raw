from PIL import Image
import cv2
import numpy as np
import requests
import asyncio
from flask import jsonify
from fastapi.responses import JSONResponse as handle_response
import datetime
from config_app.config import get_config

config_app, config_model = get_config()

def convert_pil_to_np(image):
    img = Image.open(image)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    return img

def convert_np_to_pillow(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

def handle_error(message, detailmessage,  code, detailcode):
    response = handle_response({
        'message': message,
        'detailmessage': detailmessage,
        'code': code,
        'DetailCode':detailcode
    })
    response.status_code = code
    return response


def idcard_fake_recognize(result_ocr):
    now = datetime.datetime.now()
    if result_ocr["identCardType"] =="GIẤY CHỨNG MINH NHÂN DÂN":
        if result_ocr["identCardIssueDate"].count("/")==2:
            Issue_Date = result_ocr["identCardIssueDate"].split("/")
            if len(Issue_Date[2])==4:
                ExpireDate = now.replace(day = int(Issue_Date[0]),month = int(Issue_Date[1]),  year=int(Issue_Date[2])+15)
                if ExpireDate < now:
                    return "ExpireDate"
    if result_ocr["identCardType"] == "CĂN CƯỚC CÔNG DÂN":
        # Issue_Date = result_ocr["identCardExpireDate"].split("/")

        CardNumber = result_ocr['identCardNumber']
        not_region_code = [3,5,7,9,13,16,18,21,23,28,29,32,39,41,43,47,50,53,55,57,59,61,63,65,69,71,73,76,78,81,85,88,90]
        if CardNumber == "":
            return "No cardnumber"
        else:
            region_code = CardNumber[0:3]
            if int(region_code) > 96 or int(region_code) in not_region_code:
                return "Region code not found"
            gender_code = CardNumber[3]
            year_born = result_ocr["identCardBirthDate"].split("/")[2]
            if 1900 <= int(year_born) < 2000 and int(gender_code) not in [0,1]:
                return "Gender Code is not appropriate"
            if 2000 <= int(year_born) < 2100 and int(gender_code) not in [2,3]:
                return "Gender Code is not appropriate"
            if 2100 <= int(year_born) < 2200 and int(gender_code) not in [4,5]:
                return "Gender Code is not appropriate"
            if 2200 <= int(year_born) < 2300 and int(gender_code) not in [6,7]:
                return "Gender Code is not appropriate"
            if 2300 <= int(year_born) < 2400 and int(gender_code) not in [8,9]:
                return "Gender Code is not appropriate"
            year_code = CardNumber[4:6]
            if year_born[2:] != year_code:
                return "Year Code is not appropriate"

        try:
            datetime.datetime.strptime(result_ocr["identCardExpireDate"], '%d/%m/%Y')
            Expire_Date = result_ocr["identCardExpireDate"].split("/")
            ExpireDate = now.replace(day=int(Expire_Date[0]), month=int(Expire_Date[1]), year=int(Expire_Date[2]))
            if ExpireDate < now:
                return "ExpireDate"
        except ValueError:
            try:            
                datetime.datetime.strptime(result_ocr["identCardBirthDate"], '%d/%m/%Y')
                Expire_Day = result_ocr["identCardBirthDate"].split("/")[0]
                Expire_Month = result_ocr["identCardBirthDate"].split("/")[1]
                print('Expire_Day',Expire_Day)
                try:
                    datetime.datetime.strptime(result_ocr["identCardIssueDate"], '%d/%m/%Y')
                    Age = int(result_ocr["identCardIssueDate"].split("/")[2]) - int(result_ocr["identCardBirthDate"].split("/")[2])
                    if Age <23:
                        Expire_Year = result_ocr["identCardBirthDate"].split("/")[2] + 25
                    if Age >23 and Age <38:
                        Expire_Year = result_ocr["identCardBirthDate"].split("/")[2] + 40
                    if Age >38 and Age <58:
                        Expire_Year = result_ocr["identCardBirthDate"].split("/")[2] + 60
                    else:
                        Expire_Year = result_ocr["identCardBirthDate"].split("/")[2] + 40
                    ExpireDate = now.replace(day=int(Expire_Day), month=int(Expire_Month),year=int(Expire_Year))
                    if ExpireDate < now:
                        return "ExpireDate"
                except ValueError:
                    return "ok"
            except ValueError:
                return "ok"

class StringDistance:
    '''
    Implement distance between two strings use edit distance
    '''
    def __init__(self, cost_dict_path=None):
        self.cost_dict = dict()
        if cost_dict_path is not None:
            self.load_cost_dict(cost_dict_path)

    def load_cost_dict(self, filepath):
        if self.cost_dict is None:
            self.cost_dict = dict()
        with open(filepath, encoding = 'utf-8') as f:
            for line in f:
                char1, char2, cost = line.rstrip().split('\t')
                if char1 and char2:
                    self.cost_dict[(char1, char2)] = int(cost)

    def distance(self, source, target):
        '''
        Levenshtein distance between source string and target string
        '''
        if source == target: return 0
        elif len(source) == 0: return len(target)*10
        elif len(target) == 0: return len(source)*10
        v0 = [None] * (len(target) + 1)
        v1 = [None] * (len(target) + 1)
        for i in range(len(v0)):
            v0[i] = i * 10
        for i in range(len(source)):
            v1[0] = (i + 1)*10
            for j in range(len(target)):
                cost = 0 if source[i] == target[j] else self.cost_dict.get((source[i], target[j]), 8)
                v1[j + 1] = min(v1[j] + 4 if target[j] != '.' else v1[j] + 7, v0[j + 1] + 10, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(target)]

def extract_digit(text: str):
    res = ''
    for char in text:
        if char.isdigit():
            res += char
    return res

def bounding_box(box):
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]
    return int(x1), int(y1), int(x2), int(y2)

def sort_words(bboxes):
    bboxes = sorted(bboxes, key=lambda x: x[1])
    return bboxes
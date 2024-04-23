from typing import Callable, List
from fastapi import FastAPI, Request, Response, HTTPException, Form, File, UploadFile
from fastapi.routing import APIRoute
from fastapi.exceptions import FastAPIError
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
from time import strftime
import os
import time
import traceback
import torch
import ast
from config_app.config import get_config
from utils_ocr.logging_ekyc import Logger_Days, Logger_maxBytes
from process.PretrainedModel import PretrainedModel
from service import ocr as ocr_service

if not os.path.exists("./logs"):
    os.makedirs("./logs")
file_name = './logs/logs'
log_obj = Logger_Days(file_name)

config_app, config_model = get_config()
models = PretrainedModel(config_model['ocr_model'])

app = FastAPI()
numberrequest = 0
@app.post('/ocr')
async def post(IdCardFront: UploadFile = File(...), IdCardBack: UploadFile = File(...), id_request: str = Form(...), 
               user: str = Form(...), request: Request = None):
    global numberrequest
    numberrequest = numberrequest + 1
    print("numberrequest", numberrequest)
    log_obj.info("-------------------------NEW_SESSION----------------------------------")
    log_obj.info("GuildID  = :" + " " + str(id_request))
    log_obj.info("User  = :" + " " + str(user))
    log_obj.info("numberrequest:" + " " + str(numberrequest))
    log_obj.info("path image 1:" + " " + str(IdCardFront.file))
    log_obj.info("path image 2:" + " " + str(IdCardBack.file))
    log_obj.info("ip_client: " +str(request.client.host))
    result = ocr_service.ocr(IdCardFront.file, IdCardBack.file, id_request, user, log_obj)
    return result

from service.ocr import warmup_api
warmup_api(log_obj)
uvicorn.run(app, host=config_app['server']['ip_address'], port=int(config_app['server']['port']))
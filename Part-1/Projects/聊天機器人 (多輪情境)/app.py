from __future__ import unicode_literals
import os
import requests
import json
import configparser
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, StickerSendMessage
)
import thsr_utils as Thsr
import stock_utils as Stock

app = Flask(__name__)

# LINE 聊天機器人的基本資料
config = configparser.ConfigParser()
config.read('cupoy_config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

# 接收 LINE 資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        print("body:", body)
        print("signature:", signature)
        print("===")
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 回覆設定 (defaul僅能回固定語句)
@handler.add(MessageEvent, message=TextMessage)
def get_response(event):
    query = event.message.text
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text="不論你說什麼 我都回你好!"))

thsr = Thsr.ThsrModule()

chat_record = []
thsr_res = {"starting": "", "ending": "", "date": "", "ampm": ""}
station_names = Thsr.station_name
date_keys = Thsr.date_key
ampm_keys = Thsr.ampm_key


# LINE 聊天機器人的基本資料
config = configparser.ConfigParser()
config.read('cupoy_config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

# 接收 LINE 資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        print("body:", body)
        print("signature:", signature)
        print("===")
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 回覆設定 (加入高鐵API多輪對話)
@handler.add(MessageEvent, message=TextMessage)
def get_response(event):
    query = event.message.text

    if len(chat_record) < 5:
        chat_record.append(query)
    else:
        chat_record.pop(0)
        chat_record.append(query)
    print("chat_record:", chat_record)

    private_word = ['身高', '體重', '年齡', '收入']
    if any(word in query for word in private_word):
        line_bot_api.reply_message(event.reply_token, StickerSendMessage(package_id=2, sticker_id=149))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='只能問高鐵和股票唷～'))

    # 判斷是否為"高鐵查詢意圖"
    if query == "高鐵":
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text="哪一天出發?"))
    try:
        if chat_record[-2] == "高鐵" and any(chat_record[-1] == i for i in date_keys):
            date_format = thsr.get_date_string_today(chat_record[-1])
            thsr_res['date'] = date_format
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="上午還是下午的車?"))

        elif any(chat_record[-2] == i for i in date_keys) and any(chat_record[-1] == i for i in ampm_keys):
            thsr_res['ampm'] = chat_record[-1]
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="起站是哪裡呢?"))

        elif any(chat_record[-2] == i for i in ampm_keys) and any(chat_record[-1] == i for i in station_names):
            startind_id = Thsr.station_id[chat_record[-1]]
            thsr_res['starting'] = startind_id
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text="終點站是哪裡呢?"))

        elif any(chat_record[-2] == i for i in station_names) and any(chat_record[-1] == i for i in station_names):
            ending_id = Thsr.station_id[chat_record[-1]]
            thsr_res['ending'] = ending_id

            # print("***",thsr_res['starting'],thsr_res['ending'],thsr_res['date'],thsr_res['ampm'])
            text = thsr.get_runs(
                thsr_res['starting'], thsr_res['ending'], thsr_res['date'], thsr_res['ampm'])
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text=text))

    except IndexError:
        pass

    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text="不論你說什麼 我都回你好!"))

    # 判斷是否為"股票查詢"
    if query == '股票':
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='想問哪一支股票?'))
    try:
        if '股票' in chat_record:
            symbols = Stock.stock_symbol
            if any(s == query for s in list(symbols.keys())):
                stock_symbol = symbols[query]
                stock_data = Stock.get_stockdata(stock_symbol, '2020-12-01', '2020-12-10')
                stock_info = Stock.get_stockinfo(query, stock_data, 'Close')
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=stock_info))
    except:
        pass

if __name__ == "__main__":
    app.run()

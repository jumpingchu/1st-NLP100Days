
import hmac
import base64
import datetime
from wsgiref.handlers import format_date_time
from hashlib import sha1
from time import mktime
from requests import request


# 請使用自行申請之 app_id 及 app_key
app_id = 'db8c0b51468c4361823fa5dbfd193417'
app_key = '1J4d_VxhTbejK7G1gUgIPsvYzOs'

# 車站代號對照
station_name = ['南港', '台北', '板橋', '桃園', '新竹',
                '苗栗', '台中', '彰化', '雲林', '嘉義', '台南', '左營']
station_id = {'南港': '0990', '台北': '1000', '板橋': '1010', '桃園': '1020', '新竹': '1030', '苗栗': '1035',
              '台中': '1040', '彰化': '1043', '雲林': '1047', '嘉義': '1050', '台南': '1060', '左營': '1070'}

# 乘車時間轉換
today = ["今天", "今日", "本日"]
tomorrow = ["明天", "明日"]
day_af_tomorrow = ["後天"]
days_af_tomorrow = ["大後天"]
date_key = today + tomorrow + day_af_tomorrow + days_af_tomorrow
am = ["上午", "中午以前", "中午之前", "早上", "白天"]
pm = ["下午", "中午以後", "中午之後", "晚上"]
ampm_key = am + pm

# 建立模組
class ThsrModule():

    def __init__(self):
        self.app_id = app_id
        self.app_key = app_key

    # 建立資料傳輸之hash值及完成header設定
    def get_auth_header(self):
        xdate = format_date_time(mktime(datetime.datetime.now().timetuple()))
        hashed = hmac.new(self.app_key.encode('utf8'),
                          ('x-date: ' + xdate).encode('utf8'), sha1)
        signature = base64.b64encode(hashed.digest()).decode()

        authorization = 'hmac username="' + self.app_id + '", ' + \
                        'algorithm="hmac-sha1", ' + \
                        'headers="x-date", ' + \
                        'signature="' + signature + '"'
        return {
            'Authorization': authorization,
            'x-date': format_date_time(mktime(datetime.datetime.now().timetuple())),
            'Accept - Encoding': 'gzip'
        }

    # 關鍵字判斷乘車時間
    def get_date_string_today(self, time_keywords):
        now = datetime.datetime.now()
        if any(keyword == time_keywords for keyword in today):
            pass
        elif any(keyword == time_keywords for keyword in tomorrow):
            now = now + datetime.timedelta(days=1)
        elif any(keyword == time_keywords for keyword in day_af_tomorrow):
            now = now + datetime.timedelta(days=2)
        elif any(keyword == time_keywords for keyword in days_af_tomorrow):
            now = now + datetime.timedelta(days=3)
        else:
            pass
        return now.strftime("%Y-%m-%d")

    # 取得班次資訊
    def get_runs(self, starting, ending, date, ampm):
        runs_info = request('get',
                            'https://ptx.transportdata.tw/MOTC/v2/Rail/THSR/DailyTimetable/OD/{}/to/{}/{}?$top=30&$format=JSON'.format(
                                starting, ending, date),
                            headers=self.get_auth_header())

        res_am, res_pm = [], []
        for ri in runs_info.json():
            train_no = ri["DailyTrainInfo"]["TrainNo"]
            start_time = ri["OriginStopTime"]['ArrivalTime']
            arrival_time = ri["DestinationStopTime"]['ArrivalTime']
            run = "班次:{} 的出發時間為:{} 抵達時間為:{}".format(
                train_no, start_time, arrival_time)
            if int(start_time[:2]) < 12:
                res_am.append(run)
            else:
                res_pm.append(run)

        if any(ampm == i for i in am):
            return str(res_am)
        elif any(ampm == i for i in pm):
            return str(res_pm)

        else:
            return None


if __name__ == '__main__':
    thsr = ThsrModule()
    runs = thsr.get_runs("0990", "1070", "2020-12-25", "上午")
    print(runs)

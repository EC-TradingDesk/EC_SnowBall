import json
import time
import pandas as pd
import hashlib
import requests
import random
import string
import sys, os
import urllib3
urllib3.disable_warnings()
import datetime

class Easycoins():
    def __init__(self, token=None, secret=None):
        self._host = 'https://oapi.easycoins.com'
        self._token = token
        self._secret = secret

    # http请求方法
    def _request(self, args):
        if not 'timeout' in args:
            args['timeout'] = 60

        if not 'method' in args:
            args['method'] = 'GET'
        else:
            args['method'] = args['method'].upper()

        # header设置
        if not 'headers' in args:
            args['headers'] = {}

        if not 'user-agent' in args['headers']:
            args['headers'][
                'user-agent'] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"

        # Cookies
        cookies = {}
        if 'cookies' in args:
            cookies = args['cookies']

        if not 'data' in args:
            args['data'] = {}

        args['headers'].update(self._mkHeader(args['data']))

        try:
            if args['method'] == 'GET':
                r = requests.get(args['url'], params=args['data'], cookies=cookies, headers=args['headers'],
                                 timeout=int(args['timeout']), verify=False)
            elif args['method'] == 'POST':
                r = requests.post(args['url'], data=args['data'], cookies=cookies, headers=args['headers'],
                                  timeout=int(args['timeout']), verify=False)
        except Exception as e:
            return None

        result = {}
        result['code'] = r.status_code
        ck = {}
        for cookie in r.cookies:
            ck.update({cookie.name: cookie.value})
        result['cookies'] = ck
        result['headers'] = r.headers
        result['text'] = r.text
        result['content'] = r.content
        result['raw'] = r

        return result

    # http 带签名的header生成方法
    def _mkHeader(self, data=dict()):
        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 5))
        Nonce = "%d_%s" % (int(time.time()), ran_str)
        header = dict()
        header['Token'] = self._token
        header['Nonce'] = Nonce
        header['Signature'] = self._sign(Nonce, data)

        return header

    # 签名生成方法
    def _sign(self, Nonce, data=dict()):
        tmp = list()
        tmp.append(self._token)
        tmp.append(self._secret)
        tmp.append(Nonce)
        for d, x in data.items():
            tmp.append(str(d) + "=" + str(x))
        return hashlib.sha1(''.join(sorted(tmp)).encode()).hexdigest()
    # 查询24小时行情数据
    def market_data_24h(self, symbol=None):
        args = dict()
        args['url'] = self._host + '/openApi/market/detail'
        args['data'] = dict()
        if symbol:
            args['data']['symbol'] = symbol
        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
######
    def market_kline(self, symbol=None, period=None, size=1):
        '''
            查kline
            可选参数：
            symbol: 必填，string，交易对，如BTC-USDT
            period：必填，string，如1min
            size: 非必填，int，获取数量 1-2000
        '''
        args = dict()
        args['url'] = self._host + '/openApi/market/kline'
        args['data'] = dict()
        if symbol:
            args['data']['symbol'] = symbol
            args['data']['period'] = period
            args['data']['size'] = size
        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
    # 查询市场成交
    def market_transaction(self, symbol=None, size=None):
        args = dict()
        args['url'] = self._host + '/openApi/market/trade'
        args['data'] = dict()
        if symbol:
            args['data']['symbol'] = symbol
            args['data']['size'] = size
        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
    def entrust_historylist(self, symbol = None, order_sn = None, direct = None, limit = None):
        args = dict()
        args['url'] = self._host + '/openApi/entrust/historyList'
        args['data'] = dict()
        if symbol:
            args['data']['symbol'] = symbol
            args['data']['from'] = order_sn
            args['data']['direct'] = direct
            args['data']['limit'] = limit
        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)


######
    def wallet(self, currency = None, show_all = None):
        '''
            查询账户现货币种资产
            可选参数
            currency: 币种，string，如BTC，非必填
            show_all: 是否展示全部币种，int，否则只展示有余额币种，非必填
        '''
        args = dict()
        args['url'] = self._host + '/openApi/wallet/list'
        args['data'] = dict()
        if currency:
            args['data']['currency'] = currency
        if show_all:
            args['data']['show_all'] = show_all
        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
######
    def add(self, symbol = None, type_ = None, amount = None, price = None ):
        '''
            委托挂单
            可选参数
            sympol: 必填，交易对，string, 如BTC-USDT
            type_: 必填，订单类型，string，如buy-market,buy-limit
            amount: 必填，数量，float, 注意市价买单下表示买多少钱(usdt)，市价卖单下表示卖多少币
            price: 非必填，价格,float，市价单不用此参数
        '''
        args = dict()
        args['url'] = self._host + '/openApi/entrust/add'
        args['method'] = 'POST'
        args['data'] = dict()
        args['data']['symbol'] = symbol
        args['data']['type'] = type_
        args['data']['amount'] = amount
        if price:
            args['data']['price'] = price

        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
######
    def cancel(self, order_ids = None, symbol = None ):
        '''
            撤销委托
            可选参数
            order_ids: 非必填，订单编号，string, 批量逗号分割
            symbol: 非必填，交易对，string，如BTC-USDT

        '''
        args = dict()
        args['url'] = self._host + '/openApi/entrust/cancel'
        args['method'] = 'POST'
        args['data'] = dict()
        if symbol:
            args['data']['symbol'] = symbol
        if order_ids:
            args['data']['order_ids'] = order_ids
        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
######
    def currentlist(self, symbol = None, from_ = None, direct = None, limit = None ):
        '''
            查询当前委托列表
            可选参数
            symbol: 非必填，交易对，string，如BTC-USDT，不传为所有交易对
            from_: 非必填，int, 查询其实order_sn
            direct：非必填，string，默认prev，向前
            limit：非必填，int，默认20

        '''
        args = dict()
        args['url'] = self._host + '/openApi/entrust/currentList'
        args['data'] = dict()
        if symbol:
            args['data']['symbol'] = symbol
        if from_:
            args['data']['from'] = from_
        if direct:
            args['data']['direct'] = direct
        if limit:
            args['data']['limit'] = limit


        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
####
    def status(self, order_sn = None ):
        '''
            查询委托详情
            可选参数
            order_sn：必填，string，订单编号

        '''
        args = dict()
        args['url'] = self._host + '/openApi/entrust/status'
        args['data'] = dict()
        args['order_sn'] = order_sn

        result = self._request(args)
        try:
            if result['code'] == 200:
                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)
###
    def fee(self, symbol = None ):
        '''
            查询费率
            可选参数
            symbol：必填，string，交易对，如BTC-USDT

        '''
        args = dict()
        args['url'] = self._host + '/openApi/entrust/rate'
        args['data'] = dict()
        args['symbol'] = symbol

        result = self._request(args)
        try:
            if result['code'] == 200:

                data = result['raw'].json()
                if data['errno'] == 0:
                    return data['result']
                else:
                    return data['errmsg']
            else:
                return result['raw'].status_code
        except Exception as e:
            return str(e)


token = 'f35e3f84a657d089231d51ed01bbeba5'
secret = 'hez4gkpwvuxzg06m8thn'

api = Easycoins(token,secret)
#print(api.market_kline('BTC-USDT', '1min', 10))

#api_data = api.market_kline('BTC-USDT', '1min', 10)

# api_ts = api_data['ts']
# api_symbol = api_data['symbol']
# api_period = api_data['period']
# api_series = api_data['data']

api_market_transaction = api.market_transaction('BTC-USDT',2000)
api_market_transaction_symbol = api_market_transaction['symbol']
api_market_transaction_ts = api_market_transaction['ts']
api_market_transaction_data = api_market_transaction['data']

'''df_history = pd.read_excel('历史成交记录.xlsx',sheet_name='BTC', header=0, index_col=0)

for i in range(len(api_market_transaction_data)):
    if api_market_transaction_data[i]['id'] in df_history.index:
        continue
    else:
        df_history = pd.concat([df_history, pd.DataFrame(data={'amount': api_market_transaction_data[i]['amount'],
                                                           'price': api_market_transaction_data[i]['price'],
                                                           'direction': api_market_transaction_data[i]['direction'],
                                                           'ts': api_market_transaction_data[i]['ts'],},
                                                           index=[api_market_transaction_data[i]['id']],
                                                           columns=df_history.columns)])

df_history.to_excel("历史成交记录.xlsx", sheet_name='BTC')'''
#print(api.entrust_historylist('BTC-USDT',122,'prev',20))

'''
    direct: 查询方向(默认 prev)，prev 向前，时间（或 ID）倒序；next向后，时间（或 ID）正序）。（举例一列数：1，2，3，
4，5。from=4，prev有3，2，1；next只有5）
    limit: 分页返回的结果集数量，默认为20，最大为100(具体参见分页处的描述)
'''
api_entrust_historylist = api.entrust_historylist('BTC-USDT',10,'prev',30)

df_history = pd.read_excel('历史成交记录.xlsx',sheet_name='BTC', header=0, index_col=0)

for i in range(len(api_entrust_historylist)):
    if api_entrust_historylist[i]['order_sn'] in df_history.index:
        #print("已在Excel中")
        continue
    else:
        df_history = pd.concat([df_history, pd.DataFrame(data={'symbol': api_entrust_historylist[i]['symbol'],
                                                           'ctime': api_entrust_historylist[i]['ctime'],
                                                           'type': api_entrust_historylist[i]['type'],
                                                           'side': api_entrust_historylist[i]['side'],
                                                           'price': api_entrust_historylist[i]['price'],
                                                           'number': api_entrust_historylist[i]['number'],
                                                           'total_price': api_entrust_historylist[i]['total_price'],
                                                           'deal_number': api_entrust_historylist[i]['deal_number'],
                                                           'deal_price': api_entrust_historylist[i]['deal_price'],
                                                           'status': api_entrust_historylist[i]['status']
                                                               },
                                                           index=[api_entrust_historylist[i]['order_sn']],
                                                           columns=df_history.columns)])

df_history.to_excel("历史成交记录.xlsx", sheet_name='BTC')
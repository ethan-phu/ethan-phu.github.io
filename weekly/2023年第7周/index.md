# 2023年第7周, 新的旅途

本周报分享每周所学所知所闻所感，时间范围是 `2023-02-13` 到 `2023-02-19`
会记录一些工作及生活上有意思的事情。
本周报每周日晚上发布，会同步到本人[博客](http://ethan.js.cool)中
## 见闻

### 为个人开发者实现了一套集成化的收费方案
最近一周在捣腾 RSS 订阅软件的事情，发现市面上可以主动发现独立博客，并可以订阅万物的软件很少，用的很顺滑的工具也非常的少。于是想开发软件来实现，但是服务器维护费用也是非常高的，所以想做一个可以订阅付费，但是作为个人开发者，开通支付渠道门槛非常高。于是在网上摸索了很久，找到了一个基本可行的方案，v2geek「收费5%还是有点高的」。他基本满足我的需求，但是没有提供任何开发的接口，不能和软件体系闭环。于是我采用了爬虫技术，基本实现了平台和软件的闭环，可以达到在软件上支付订阅费用，立马给订阅者分发授权码!一整套操作，不需要任何人工操作。
```python
#coding=utf-8
import requests
import re
from bs4 import BeautifulSoup
import uuid

class V2GeeKPay(object):
    def __init__(self, cookies, project=""):
        self.cookies = cookies
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9,und;q=0.8,zh-Hans;q=0.7,en;q=0.6",
            "cache-control": "max-age=0",
            "cookie": "_sessions={}".format(cookies),
            "referer": "https://v2geek.com/udmin/projects",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "Linux",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
        }
        flag, self.project_list, self.user_name = self.get_project_list()
        if not flag:
            raise ValueError("cookie过期，请检查")
        if project != "":
            self.set_project(project)

    def set_project(self, project):
        project_exist = False
        for project_obj in self.project_list:
            if project_obj["key"] == project:
                self.pay_js = "https://v2geek.com/{}/{}/widget.js".format(self.user_name, project)
                self.login_url = "https://v2geek.com/users/sign_in"
                self.import_url = "https://v2geek.com/udmin/projects/{}/licenses/import".format(project)
                self.project_url = "https://v2geek.com/udmin/projects/{}".format(project)
                self.order_url = "https://v2geek.com/udmin/orders"
                self.project = project
                project_exist = True
                break
        if not project_exist:
            raise ValueError("当前用户不存在此项目，请检查")

    def get_project_list(self):
        """获取当前用户的项目以及用户名"""
        project_list_url = "https://v2geek.com/udmin/projects"
        session = requests.Session()
        try:
            response = str(session.get(project_list_url, headers=self.headers, verify=False).content, encoding="utf-8")
            html_obj = BeautifulSoup(response, "html.parser")
            tr_list = html_obj.select("tbody")[0].select("tr")
            dict_project_info = {
                0: "name",
                1: "price"
            }
            user_name = ""
            project_list = []
            for tr in tr_list:
                temp = {}
                for idx, td in enumerate(tr.select("td")):
                    if idx in dict_project_info:
                        pro_key = dict_project_info[idx]
                        temp[pro_key] = td.text.strip().replace("￥","").replace("元","")
                        if pro_key == "name":
                            key_list = td.select("a")[0].get("href").split("/")
                            if user_name != "":
                                user_name = key_list[1]
                            temp["key"] = key_list[2]
                project_list.append(temp)
            return True, project_list, user_name
        except:
            return False, [], ""

    def generateAuthToken(self, numbers=100):
        """使用uuid算法自动生成authToken
        """
        return [uuid.uuid4().hex for i in range(numbers)]

    def import_auth_token(self, auth_token_list):
        # 自动生成一系列的auth_token值，并分发到系统以便不断的扩充auth_token的值，并将值存入系统
        # 获取token值
        saved_token_list, unused_token, rest_token = self.get_auth_token_list()
        if rest_token != "":
            data = {
                "utf8": (None, "&#x2713;"),
                "authenticity_token": (None, rest_token),
                "licenses": (None, "\n".join(auth_token_list)),
                "commit": (None, "提交")
            }
            session = requests.Session()
            session.post(self.import_url, data=data, headers=headers, verify=False)
            print("导入成功")
            return True, "success"
        else:
            print("cookie过期")
            return False, "cookie已过期，请联系工程师处理"

    def get_auth_token_list(self):
        """
        获取账户的相关信息，以便能够实时进行激活处理
        获取成功则返回当前的token,可以用于导入auth_token
        """
        session = requests.Session()
        try:
            response = str(session.get(self.project_url, headers=self.headers, verify=False).content, encoding="utf-8")
            html_obj_first = BeautifulSoup(response, "html.parser")
            page = html_obj_first.select(".pagination")
            meta_list = html_obj_first.select("meta")
            rest_token = ""
            html_obj_list = [html_obj_first]
            for meta in meta_list:
                if meta.get("name") == "csrf-token":
                    rest_token = meta.get("content")
            if len(page):
                page_num = int(page[0].select("a")[-1].get("href").split("=")[1].strip())+1
                for i in range(2, page_num):
                    response = str(session.get(self.project_url+"?page={}".format(i), headers=headers, verify=False).content, encoding="utf-8")
                    html_obj_copy = BeautifulSoup(response, "html.parser")
                    html_obj_list.append(html_obj_copy)
            dict_order_info = {
                0: "auth_token",
                1: "use_status",
                2: "order_id",
            }
            auth_token_list = []
            temp = {}
            for html_obj in html_obj_list:
                for tr in html_obj.select("tbody")[0].select("tr"):
                    if "license" in tr.get("id"):
                        temp = {
                            "license_id": tr.get("id")
                        }
                        for idx, td in enumerate(tr.select("td")):
                            key_id = idx % 4
                            if key_id in dict_order_info:
                                temp[dict_order_info[key_id]] = td.text.strip()
                        auth_token_list.append(temp)
            # 计算还剩余下多少个token没有被分配完,这里不进行自动生成分配，交给别的业务系统来处理这个事情
            unused_token = []
            for auth_token in auth_token_list:
                if auth_token.get("order_id") == "-":
                    unused_token.append(auth_token)
            return auth_token_list, unused_token, rest_token
        except Exception as e:
            return [], [], ""

    def verify_order_token(self, order_id):
        """验证订单号有没有被分配授权码，如果有则返回True和授权码
        """
        auth_token_list, unused_token, rest_token = self.get_auth_token_list()
        for auth_token in auth_token_list:
            if auth_token.get("order_id") == order_id:
                return True, auth_token.get("auth_token")
        return False,""

    def query_order_status(self, order_id):
        """查询当前订单状态"""
        headers = { 'content-type': 'application/json','User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'}
        order_status_url = "https://v2geek.com/{user_name}/{project}/checkouts/{order_id}/status".format(user_name=self.user_name, project=self.project, order_id=order_id)
        session = requests.Session()
        res = session.get(order_status_url, headers=headers, verify=False)
        html_content = str(res.content, encoding='utf8')
        return html_content.strip()

    def query_order_list(self):
        """查询订单列表
        """
        session = requests.Session()
        res = session.get(self.order_url, headers=self.headers, verify=False)
        html_content = str(res.content, encoding='utf8')
        html_obj_first = BeautifulSoup(html_content, "html.parser")
        try:
            meta_list = html_obj_first.select("meta")
            rest_token = ""
            html_obj_list = [html_obj_first]
            for meta in meta_list:
                if meta.get("name") == "csrf-token":
                    rest_token = meta.get("content")
            html_obj_list = [html_obj_first]
            page = html_obj_first.select(".pagination")
            if len(page):
                page_num = int(page[0].select("a")[-1].get("href").split("=")[1].strip())+1
                for i in range(2, page_num):
                    response = str(session.get(self.order_url+"?page={}".format(i), headers=self.headers, verify=False).content, encoding="utf-8")
                    html_obj_copy = BeautifulSoup(response, "html.parser")
                    html_obj_list.append(html_obj_copy)
            
            dict_order_info = {
                1: "order_status",
                2: "order_id",
                3: "pay_num",
                4: "real_pay",
                5: "order_time",
                6: "pay_time",
                7: "email"
            }
            order_list = []
            temp = {}
            for html_obj in html_obj_list:
                for tr in html_obj.select("tbody")[0].select("tr"):
                    temp = {}
                    for idx, td in enumerate(tr.select("td")):
                        key_id = idx
                        if key_id in dict_order_info:
                            temp[dict_order_info[key_id]] = td.text.strip().replace("￥","").replace("元","")
                    order_list.append(temp)
            return order_list, rest_token
        except Exception as e:
            return [], ""

    def clear_expired_order(self):
        order_list, rest_token = self.query_order_list()
        if rest_token != "":
            for order in order_list:
                if order["order_status"] == "未支付" or order["order_status"] == "已关闭":
                    print(self.close_order(order["order_id"]))
            return True, "success"
        else:
            return False,"cookie过期"

    def close_order(self, order_id):
        order_close_url = "https://v2geek.com/udmin/orders/{}/close".format(order_id)
        order_list, rest_token = self.query_order_list()
        if rest_token != "":
            # 查看当前订单状态
            for order in order_list:
                if order["order_id"] == order_id:
                    if order["order_status"] == "已关闭":
                        self.delete_order(order_id, rest_token)
                        return True
                    elif order["order_status"] == "未支付":
                        session = requests.Session()
                        data = {
                            "_method": (None, "post"),
                            "authenticity_token": (None, rest_token),
                        }
                        res= session.post(order_close_url, data=data, headers=self.headers, verify=False)
                        close_content = str(res.content, encoding='utf8')
                        html_obj = BeautifulSoup(close_content, "html.parser")
                        meta_list = html_obj.select("meta")
                        for meta in meta_list:
                            if meta.get("name") == "csrf-token":
                                rest_token = meta.get("content")
                                if rest_token:
                                    self.delete_order(order_id, rest_token)
                                    return True, "success"
                                else:
                                    return False, "关闭失败"
                            else:
                                return False, "关闭失败"
                    else:
                        return False, "订单为{}状态,不支持关闭".format(order["order_status"])
        else:
            return False,"cookie过期"

    def delete_order(self, order_id, token):
        delete_url = "https://v2geek.com/udmin/orders/{}".format(order_id)
        data = {
            "_method": (None, "delete"),
            "authenticity_token": (None, token),
        }
        session = requests.Session()
        session.post(delete_url, data=data, headers=self.headers, verify=False)

    def moke_pay(self, email, coupon=None):
        """.dockerignore"""
        headers = { 'content-type': 'application/json','User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'}
        session = requests.Session()
        res = session.get(self.pay_js, headers=headers, verify=False)
        js = str(res.content, encoding='utf8')
        def get_pay_token(js):
            """.dockerignore"""
            it = re.finditer(r"value=([\s\S]*?)/>", js)
            try:
                for match in it:
                    ret = match.group()
                    token = ret.split("\"")[1].replace('\\', "")
                    if len(token) > 16:
                        return token
            except:
                raise ValueError("代码格式错误了，请联系工程师进行修复")
        
        def get_order_id(content):
            """.dockerignore"""
            it = re.finditer(r"/checkouts/([\s\S]*?)/status", content.decode("utf-8"))
            try:
                for match in it:
                    ret = match.group()
                    return ret.split("/")[2]
            except:
                raise ValueError("获取订单，代码格式错误，请联系工程师进行修复")
        
        def get_pay_url(js):
            """.dockerignore"""
            it = re.finditer(r"action=([\s\S]*?)/>", js)
            try:
                for match in it:
                    ret = match.group()
                    reg = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                    url_list = re.findall(reg, ret)
                    for url in url_list:
                        if len(url) > 16:
                            return url.replace('\\', "")
            except Exception as e:
                raise ValueError("代码格式错误，请联系工程师进行修复")

        token = get_pay_token(js)
        pay_url = get_pay_url(js)
        data = {
            "utf8": (None, "&#x2713;"),
            "authenticity_token": (None, token),
            "order[buyer_email]": (None, email),
            "order[coupon]": (None, coupon),
            "order[payment]": (None, "wechat"),
            "commit": (None, "提交")
        }
        try:
            r = session.post(pay_url, data=data, verify=False)
            pay_content = r.content
            order_id = get_order_id(pay_content)
            html_obj = BeautifulSoup(pay_content, "html.parser")
            back_url = "https://v2geek.com/{}/{}/checkouts/{}/paying".format(self.user_name, self.project, order_id)
            return order_id, html_obj.select('.qrcode')[0].get('src'), back_url
        except:
            raise ValueError("代码格式错误，请联系工程师进行修复")
```

### 开发一款RSS订阅软件
在这信息爆炸的时代，我们身边围绕着十分多的信息，加上推荐系统不断在为我们构建信息围城，我们需要寻找更好的阅读体验的方式，去打破信息围城，以突破信息边界，让我们和世界的信息差不断缩小。有时候也在迷茫，想寻找更好的资讯渠道，比如优质博主，优质信息。知识付费盛行，导致优质信息获取进一步变得十分困难。所以内心一直在想做一款能够对内容进行涮选的rss订阅软件，来丰富自己的阅读，并提升阅读体验和更快的找到优质的信息。

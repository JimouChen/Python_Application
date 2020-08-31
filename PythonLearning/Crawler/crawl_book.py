"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import requests
import re
import pandas


def get_html_text(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    headers = {'User-Agent': user_agent}
    # 发送请求
    respond = requests.get(url=url, headers=headers)
    respond.encoding = 'GBK'
    html_text = respond.text

    return html_text


def match_text(html_text):
    # article = []
    # 需要得到的内容，在  .+?  两边一定要加括号
    pattern = re.compile(r'<p>(.+?)</p>')
    msg = pattern.findall(html_text)[0]

    # 得到文章标题
    pattern2 = re.compile(r'<p id="bktitle">(.+?)</p>')
    title = pattern2.findall(html_text)[0]
    # article.append({'标题': title, '内容': msg})
    # article.append(title)
    # article.append(msg)
    # return article
    return title + '\n' + msg + '\n'


def next_url(html_text):
    pattern = re.compile(r">下一篇：<a href='(.+?)'>第")
    next_page = pattern.findall(html_text)
    return next_page


if __name__ == '__main__':

    all_content = []
    url = 'http://www.newxue.com/baike/12454027234683.html'
    # html_text = get_html_text(url)
    # massage = match_text(html_text)

    # 保存成txt
    file = open('SGYY_book.txt', 'w')

    while url:
        try:
            html_text = get_html_text(url)
            massage = match_text(html_text)
            print(massage)
            file.write(massage)
            all_content.append(massage)
            url = next_url(html_text)[0]
        except Exception:
            print('爬取完毕')
            break

    print(all_content)
    # save = pandas.DataFrame(all_content)
    # save.to_excel('SGYY_book.xls')
    file.close()

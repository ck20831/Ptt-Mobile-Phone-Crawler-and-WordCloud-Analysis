import requests
import re
import numpy as np
import matplotlib.pyplot as plt
import jieba.analyse
from bs4 import BeautifulSoup
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
from scipy.ndimage import gaussian_gradient_magnitude

# 基本參數
url = "https://www.ptt.cc/bbs/MobileComm/search?q=30k"
payload = {
    'from': '/bbs/Gossiping/index.html',
    'yes': 'yes'
}
data = []  # 全部文章的資料
num = 0
all_content = []

# 用session紀錄此次使用的cookie
rs = requests.session()
response = rs.post("https://www.ptt.cc/ask/over18", data=payload)

# 爬取兩頁
for i in range(2):
    # get取得頁面的HTML
    # print(url)
    response = rs.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # print(soup.prettify())

    # 找出每篇文章的連結
    links = soup.find_all("div", class_="title")
    for link in links:
        # 如果文章已被刪除，連結為None
        if link.a != None:

            article_data = {}  # 單篇文章的資料
            page_url = "https://www.ptt.cc/" + link.a["href"]

            # 進入文章頁面
            response = rs.get(page_url)
            result = BeautifulSoup(response.text, "html.parser")
            # print(soup.prettify())

            # 找出作者、標題、時間、留言
            main_content = result.find("div", id="main-content")
            article_info = main_content.find_all(
                "span", class_="article-meta-value")

            if len(article_info) != 0:
                title = article_info[2].string  # 標題
            else:
                title = "無"  # 標題

            # 將整段文字內容抓出來
            all_text = main_content.text
            # 以--切割，抓最後一個--前的所有內容
            pre_texts = all_text.split("--")[:-1]
            # 將前面的所有內容合併成一個
            one_text = "--".join(pre_texts)
            # 以\n切割，第一行標題不要
            texts = one_text.split("\n")[1:]
            # 將每一行合併
            content = "\n".join(texts)

            content = content.replace("─", "n") \
                .replace("例：10K-20K/iPhone12 Pro Max、Note20U、XPERIA 5II", "") \
                .replace("(螢幕尺寸/拍照/效能/續航力/防水/記憶卡...等)：", "") \
                .replace("大家好，小弟最近想更換手機，所以想拜託各位幫小弟推薦一下機種：", "") \
                .replace("ipad", '') \
                .replace("NEO", '')

            content_1 = re.sub("─|\\|─|", "", content)
            content_2 = content_1.replace("：", "分割").replace("2.使用需求", "分割").replace("3.品牌喜好","分割").split("分割")
            # print(content_2)
            while '' in content_2:
                content_2.remove('')

            # print(title)
            for keyword in ['手錶', '平板', '手寫筆', '左右', 'Re', '不拍照，只玩遊戲android手機推薦', '20K內的安卓拍照機?']:
                if keyword in title:
                    break
            else:
                article_data["title"] = title
                article_data["content0"] = content_2[0].replace("\n", "")
                article_data["content1"] = content_2[1].replace("\n", "")
                article_data["content2"] = content_2[2].replace("\n", "")
                all_content.append(content_2[2]
                                   .replace("\n", "")
                                   .replace("3.品牌喜好", "")
                                   .replace("3", '')
                                   .replace("B", "")
                                   .replace("C", "")
                                   .replace("4", ""))
            data.append(article_data)
            num += 1
            print("第 " + str(num) + " 篇文章完成!")

    # 找到上頁的網址，並更新url
    url = "https://www.ptt.cc/" + soup.find("a", string="‹ 上頁")["href"]

text = "".join(all_content)

dictfile = "dict.txt"  # 字典檔
stopfile = "stopwords.txt"  # stopwords
fontpath = "NotoSansTC-Regular.otf"  # 字型檔

mdfile = "".join(all_content)  # 文檔

mask_color = np.array(Image.open('parrot-by-jose-mari-gimenez2.jpg'))
mask_color = mask_color[::3, ::3]
mask_image = mask_color.copy()
mask_image[mask_image.sum(axis=2) == 0] = 255

jieba.set_dictionary(dictfile)
jieba.load_userdict("userdict.txt")
jieba.analyse.set_stop_words(stopfile)

text = "".join(all_content)

tags = jieba.analyse.extract_tags(text, topK=10)

seg_list = jieba.lcut(text, cut_all=False)
dictionary = Counter(seg_list)

freq = {}
for ele in dictionary:
    if ele in tags:
        freq[ele] = dictionary[ele]
print(freq)  # 計算出現的次數

edges = np.mean([gaussian_gradient_magnitude(mask_color[:, :, i] / 255., 2) for i in range(3)], axis=0)
mask_image[edges > .08] = 255

wordcloud = WordCloud(max_words=2000,
                      background_color='white',
                      mask=mask_image,
                      font_path=fontpath,
                      contour_width=1,
                      max_font_size=60,
                      random_state=42,
                      relative_scaling=0)

wordcloud.generate(text)

image_colors = ImageColorGenerator(mask_color)
wordcloud.recolor(color_func=image_colors)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file('30k文字雲.png')

# encoding=utf-8
import io
import json

if __name__ == "__main__":
    with io.open("data/ss.txt", encoding="utf-8") as f:
        print(type(f))
        for line in f:

            s = "{\"id\":824,\"status\":\"publish\",\"type\":\"restaurant\",\"date\":\"2017-08-24T13:48:26\",\"title\":\"\\u94b1\\u6c5f\\u65b0\\u57ce\\u4e07\\u6021\\u9152\\u5e97\\u5496\\u5561\\u5ba4\",\"link\":\"https:\\/\\/www.foodieguild.com\\/restaurant\\/824.html\",\"content\":{\"rendered\":\"\",\"protected\":\"\"},\"restaurant_score\":\"2.9\",\"restaurant_score_taste\":\"2.9\",\"restaurant_score_environment\":\"3.2\",\"restaurant_score_worth\":\"1\",\"restaurant_address\":\"\\u676d\\u5dde\\u5e02 \\u6c5f\\u5e72\\u533a \\u5bcc\\u6625\\u8def298\\u53f7\\u676d\\u5dde\\u94b1\\u6c5f\\u65b0\\u57ce\\u4e07\\u6021\\u9152\\u5e972\\u5c42\",\"restaurant_latlng\":\"30.240300,120.206640\",\"restaurant_traffic\":\"\",\"restaurant_price\":\"230\",\"restaurant_price_2\":\"\",\"restaurant_tel\":\"0571-88378888\",\"restaurant_tel2\":\"\",\"restaurant_tel3\":\"\",\"restaurant_website\":\"\",\"restaurant_seats\":\"\",\"restaurant_rooms\":\"\",\"restaurant_rooms_spec\":\"10\",\"restaurant_startyear\":\"\",\"restaurant_openhours\":\"\\u5468\\u4e00-\\u5468\\u65e5 6:30-10:00  11:00-13:30 17:00-21:00\",\"restaurant_opendays\":\"\\u4e00,\\u4e8c,\\u4e09,\\u56db,\\u4e94,\\u516d,\\u65e5\",\"restaurant_tag_newopen\":\"\",\"restaurant_tag_branch\":\"\",\"restaurant_tag_recommand\":\"\",\"restaurant_others\":\"\",\"restaurant_paymethod\":\"cash,card\",\"restaurant_photos\":[{\"url\":\"https:\\/\\/www.foodieguild.com\\/wp-content\\/uploads\\/2017\\/08\\/20170824133904.jpg\",\"des\":\"\"},{\"url\":\"https:\\/\\/www.foodieguild.com\\/wp-content\\/uploads\\/2017\\/08\\/20170824133945.jpg\",\"des\":\"\"},{\"url\":\"https:\\/\\/www.foodieguild.com\\/wp-content\\/uploads\\/2017\\/08\\/20170824133952.gif\",\"des\":\"\"},{\"url\":\"https:\\/\\/www.foodieguild.com\\/wp-content\\/uploads\\/2017\\/08\\/20170824133959.gif\",\"des\":\"\"},{\"url\":\"https:\\/\\/www.foodieguild.com\\/wp-content\\/uploads\\/2017\\/08\\/20170824134015.jpg\",\"des\":\"\"}],\"restaurant_location\":[{\"name\":\"\\u676d\\u5dde\",\"term_id\":48},{\"name\":\"\\u6c5f\\u5e72\\u533a\",\"term_id\":74}],\"restaurant_cuisine\":[{\"name\":\"\\u81ea\\u52a9\\u9910\",\"term_id\":57}],\"restaurant_time\":[{\"name\":\"\\u5348\\u9910\",\"term_id\":62},{\"name\":\"\\u65e9\\u9910\",\"term_id\":61},{\"name\":\"\\u665a\\u9910\",\"term_id\":63}],\"restaurant_feature\":[{\"name\":\"Wi-Fi\",\"term_id\":4,\"slug\":\"wifi\"},{\"name\":\"\\u53ef\\u9884\\u8ba2\",\"term_id\":7,\"slug\":\"reservation\"},{\"name\":\"\\u666f\\u89c2\\u5ea7\",\"term_id\":5,\"slug\":\"landscape-seat\"},{\"name\":\"\\u6cca\\u8f66\\u4f4d\",\"term_id\":3,\"slug\":\"park\"}],\"restaurant_thumb\":\"https:\\/\\/cdn.foodieguild.com\\/wp-content\\/uploads\\/2017\\/08\\/20170824133945-640x358.jpg\",\"restaurant_review\":\"2008\",\"restaurant_review_title\":\"\\u676d\\u57ce\\u7a81\\u7136\\u6d41\\u884c\\u5357\\u6d0b\\u83dc\\u4e86\\uff1f\\u5c0f\\u98df\\u94fa\\u7530\\u9e21\\u7ca5\\u548c\\u5927\\u9152\\u5e97\\u6d77\\u5357\\u9e21\\u996d\\u6d4b\\u8bc4\",\"restaurant_add\":null}"

            line = eval(line)
            print(line)
            print(type(json.loads(line)))
            print(type(json.loads(s)))

    with io.open("sss.txt", "w", encoding="utf-8") as f:
        list
        f.write("\t".join(list))
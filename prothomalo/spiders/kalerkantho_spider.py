import scrapy
from urllib.parse import urlparse
from datetime import datetime
from googletrans import Translator  # pip install googletrans==4.0.0rc1


class KalerKanthoSpider(scrapy.Spider):
    name = "kalerkantho"
    allowed_domains = ["kalerkantho.com"]
    start_urls = ["https://www.kalerkantho.com/"]
    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
    }
    translator = Translator()
    DIGIT_MAP = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

    def parse(self, response):
        # Collect article links
        articles = response.css("a::attr(href)").getall()
        print("no of articles",len(articles))
        for href in articles:
            if href and any(x in href for x in ["/online/", "/print/"]):
                yield response.follow(href, headers=self.custom_headers, callback=self.parse_article)

    def parse_article(self, response):
        # ---- Category ----
        category = (
            response.css('meta[property="article:section"]::attr(content)').get()
            or response.css('meta[name="section"]::attr(content)').get()
        )
        if not category:
            parts = urlparse(response.url).path.strip("/").split("/")
            if len(parts) > 1:
                category = parts[1] if parts[0] == "online" else parts[0]

        # ---- Category in Bengali ----
        category_bn = None
        if category:
            try:
                category_bn = self.translator.translate(category, dest="bn").text
            except Exception:
                category_bn = category

        # ---- Title ----
        title = response.css("h1.my-text::text").get()
        print(tilte)
        if not title:
            title = response.css('meta[property="og:title"]::attr(content)').get()
        if title:
            title = title.strip()

        # ---- Lead Image ----
        og_img = response.css('meta[property="og:image"]::attr(content)').get()
        article_image = og_img.split("?")[0] if og_img else None

        # ---- Published Date ----
        published_iso = response.css("time::attr(datetime)").get()
        if not published_iso:
            # Some Kalerkantho pages use <span class="time">text</span>
            published_iso = response.css("span.time::text").get()

        published_date_bn = None
        if published_iso:
            try:
                dt = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
                formatted = dt.strftime("%d.%m.%Y")
                published_date_bn = formatted.translate(self.DIGIT_MAP)
            except Exception:
                published_date_bn = published_iso.translate(self.DIGIT_MAP)

        # ---- Article Body ----
        paras = response.css("div.news-content p::text").getall()
        if not paras:
            paras = response.css("article p::text").getall()
        article_body = " ".join(p.strip() for p in paras if p.strip())

        yield {
            "article_title": title,
            "url": response.url,
            "article_image": article_image,
            "published_date": published_iso,
            "published_date_bn": published_date_bn,
            "article_body": article_body,
            "category": category,
            "category_bn": category_bn,
            "source": "কালের কণ্ঠ",
            "domain": "kalerkantho"
        }

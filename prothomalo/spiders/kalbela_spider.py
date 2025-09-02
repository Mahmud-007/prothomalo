import scrapy
from urllib.parse import urlparse
from datetime import datetime
from googletrans import Translator  # pip install googletrans==4.0.0rc1


class KalbelaSpider(scrapy.Spider):
    name = "kalbela"
    allowed_domains = ["kalbela.com"]
    start_urls = ["https://www.kalbela.com/"]

    translator = Translator()

    DIGIT_MAP = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

    def parse(self, response):
        for href in response.css("a::attr(href)").getall():
            if href and any(x in href for x in ["/news/", "/sports/", "/entertainment/", "/bangladesh/"]):
                yield response.follow(href, callback=self.parse_article)


    def parse_article(self, response):
        # ---- Category ----
        category = (
            response.css('meta[property="article:section"]::attr(content)').get()
            or response.css('meta[name="section"]::attr(content)').get()
        )
        if not category:
            parts = urlparse(response.url).path.strip("/").split("/")
            if len(parts) > 1:
                category = parts[0]

        # ---- Category in Bengali ----
        category_bn = None
        if category:
            try:
                category_bn = self.translator.translate(category, dest="bn").text
            except Exception:
                category_bn = category

        # ---- Title ----
        title = response.css("h1::text").get()
        if not title:
            title = response.css('meta[property="og:title"]::attr(content)').get()
        if title:
            title = title.strip()

        # ---- Lead Image ----
        og_img = response.css('meta[property="og:image"]::attr(content)').get()
        article_image = og_img.split("?")[0] if og_img else None

        # ---- Published Date (ISO → Bangla digits) ----
        published_iso = datetime.utcnow().isoformat()
        published_date_bn = None
        if published_iso:
            try:
                dt = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
                formatted = dt.strftime("%d.%m.%Y")
                published_date_bn = formatted.translate(self.DIGIT_MAP)
            except Exception:
                published_date_bn = published_iso.translate(self.DIGIT_MAP)

        # ---- Article Body ----
        paras = response.css("div.dtl_content_section p::text").getall()
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
            "source": "কালবেলা",
            "domain": "kalbela"
        }

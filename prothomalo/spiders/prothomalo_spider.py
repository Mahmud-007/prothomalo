import scrapy
from urllib.parse import urlparse
from datetime import datetime
from googletrans import Translator  # pip install googletrans==4.0.0rc1

class ProthomAloSpider(scrapy.Spider):
    name = "prothomalo"
    allowed_domains = ["prothomalo.com"]
    start_urls = ["https://www.prothomalo.com/"]

    translator = Translator()  # Google Translate client

    # English digit → Bangla digit mapping
    DIGIT_MAP = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

    def parse(self, response):
        # Find all article links
        for href in response.css("a.title-link::attr(href)").getall():
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

        # ---- Category (Bengali) ----
        category_bn = None
        if category:
            try:
                category_bn = self.translator.translate(category, dest="bn").text
            except Exception:
                category_bn = category

        # ---- Title Extraction (multi-strategy) ----
        # Strategy 1: Try span.tilte-no-link-parent full text
        title = response.xpath('//span[contains(@class, "tilte-no-link-parent")]//text()').getall()
        title = ''.join(t.strip() for t in title).strip()

        # Strategy 2: Fallback to h1 (common in articles)
        if not title:
            title = response.css("h1::text").get()
            if title:
                title = title.strip()

        # Strategy 3: Fallback to meta og:title
        if not title:
            title = response.css('meta[property="og:title"]::attr(content)').get()
            if title:
                title = title.strip()

        # ---- Lead image ----
        og_img = response.css('meta[property="og:image"]::attr(content)').get()
        article_image = og_img.split("?")[0] if og_img else None

        # ---- Published date → Bangla digits ----
        published_iso = response.css("time::attr(datetime)").get()
        published_date_bn = None
        if published_iso:
            try:
                dt = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
                formatted = dt.strftime("%d.%m.%Y")
                published_date_bn = formatted.translate(self.DIGIT_MAP)
            except Exception:
                published_date_bn = published_iso.translate(self.DIGIT_MAP)

        # ---- Article Body ----
        paras = response.css("div.story-element p::text").getall()
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
            "source": "প্রথম আলো",
            "domain": "prothomalo"
        }
# scrapy crawl prothomalo -o ./csv/prothomalo_090920251104.csv
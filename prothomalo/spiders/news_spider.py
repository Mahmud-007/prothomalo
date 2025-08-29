import scrapy

class ProthomAloSpider(scrapy.Spider):
    name = "prothomalo"
    allowed_domains = ["prothomalo.com"]
    start_urls = ["https://www.prothomalo.com/"]

    def parse(self, response):
        # Select article links on homepage
        for article in response.css("a.title-link::attr(href)").getall():
            # follow links to article pages
            yield response.follow(article, callback=self.parse_article)

    def parse_article(self, response):
        yield {
            "title": response.css("h1::text").get(),
            "url": response.url,
            "article_image": response.css('meta[property="og:image"]::attr(content)').get().split("?")[0],
            "published_date": response.css("time::attr(datetime)").get(),
            "article_body": " ".join(response.css("div.story-element p::text").getall())
        }

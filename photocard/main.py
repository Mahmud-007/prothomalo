from photocard.config import Config
from photocard.pipeline.photocard_pipeline import PhotocardPipeline

def main():
    cfg = Config()
    PhotocardPipeline(cfg).run()

if __name__ == "__main__":
    main()

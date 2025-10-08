import atexit
from typing import Optional
from playwright.sync_api import sync_playwright, Browser

class BrowserManager:
    """Lifecycle manager for a single Chromium instance."""
    _browser: Optional[Browser] = None
    _ctx = None

    def get(self) -> Browser:
        if self._browser is None:
            self._ctx = sync_playwright().start()
            self._browser = self._ctx.chromium.launch()
            atexit.register(self._shutdown)
        return self._browser

    def _shutdown(self):
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._ctx:
                self._ctx.stop()
        except Exception:
            pass
        self._browser = None
        self._ctx = None

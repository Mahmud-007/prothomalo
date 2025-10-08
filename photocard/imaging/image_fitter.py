import cv2

class ImageFitter:
    """Aspect-fill 'cover' fit and center-crop to target size."""
    @staticmethod
    def fit_cover(src_bgr, tw: int, th: int):
        sh, sw = src_bgr.shape[:2]
        src_as = sw / sh
        tgt_as = tw / th
        if src_as < tgt_as:
            new_w, new_h = tw, int(tw / src_as)
        else:
            new_h, new_w = th, int(th * src_as)
        resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = (new_w - tw) // 2
        y = (new_h - th) // 2
        return resized[y:y + th, x:x + tw]

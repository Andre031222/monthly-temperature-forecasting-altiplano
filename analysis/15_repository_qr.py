import qrcode
from qrcode.constants import ERROR_CORRECT_M

from config import FIGURES

REPOSITORY = "https://github.com/Andre031222/monthly-temperature-forecasting-altiplano"


def build_qr(url, filename, box_size=10, border=2):
    code = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M,
                         box_size=box_size, border=border)
    code.add_data(url)
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white").convert("RGB")

    for suffix in ("png", "tiff"):
        path = FIGURES / f"{filename}.{suffix}"
        image.save(path, dpi=(300, 300))
    return image.size


def main():
    size = build_qr(REPOSITORY, "repository_qr")
    print("url :", REPOSITORY)
    print("size:", size, "px")
    for suffix in ("png", "tiff"):
        path = FIGURES / f"repository_qr.{suffix}"
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

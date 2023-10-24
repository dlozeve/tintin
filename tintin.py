import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, feature, morphology, measure
from tqdm import tqdm

IMAGES_DIR = Path("images")


def main():
    albums = sorted(list(Path("albums").iterdir()))
    for album in albums:
        album_dir = extract_pages(album)
        process_album(album_dir)


def extract_pages(album: Path) -> Path:
    album_name = album.stem
    print("Extracting pages of", album_name)
    album_dir = IMAGES_DIR / album_name
    if (album_dir / "pages").is_dir():
        return album_dir
    (album_dir / "pages").mkdir(parents=True)
    # /etc/ImageMagick-6/policy.xml: remove ghostscript limitations and bump disk cache size to 16GiB
    subprocess.run(
        ["convert", "-density", "300", album.as_posix(), album_dir / "pages" / "page.png"]
    )
    return album_dir


def process_album(album_dir: Path):
    page_files = list((album_dir / "pages").glob("page-*.png"))
    for page_file in tqdm(page_files, desc="Extracting panels of " + album_dir.name):
        page_idx = int(page_file.stem.split("-")[1])
        im = Image.open(page_file)
        page_dir = album_dir / f"page{page_idx:02d}"
        if page_dir.is_dir():
            continue
        page_dir.mkdir()
        panels = get_panels(im)  # type: ignore
        for panel_idx, panel in enumerate(panels):
            panel.save(page_dir / f"panel{panel_idx:02d}.png")  # type: ignore


def get_panels(im: Image) -> list[Image]:  # type: ignore
    # https://maxhalford.github.io/blog/comic-book-panel-segmentation/
    grayscale = color.rgb2gray(color.rgba2rgb(im))
    edges = feature.canny(grayscale)
    thick_edges = morphology.dilation(edges)
    segmentation = ndi.binary_fill_holes(thick_edges)
    labels = measure.label(segmentation)

    panels = []
    for label in np.unique(labels):
        if label == 0:  # background
            continue
        xs, ys = np.where(labels == label)
        left, upper, right, lower = np.min(ys), np.min(xs), np.max(ys), np.max(xs)
        if right - left < 100 or lower - upper < 100:
            continue
        panels.append(im.crop((left, upper, right, lower)))  # type: ignore

    return panels


if __name__ == "__main__":
    main()

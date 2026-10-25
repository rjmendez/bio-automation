"""darkroom: the storefront listing, the similarity match and the sync that writes prints.csv.

The network and the GPU are the injected dependencies: a FakeSite serves fixed HTML (and refuses any URL it was not given), the product
download returns fixed bytes, and the embedder maps an image's bytes to a fixed vector, so a product "looks like" the gallery file whose
bytes it was given. Gallery files and every output are real files under tmp_path."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from bio_automation import darkroom
from bio_automation.darkroom import DarkroomError, Product

BASE = "https://shop.test"


def card(pid, collection=None, img=True):
    path = f"/products/{collection}/{pid}" if collection else f"/products/{pid}"
    pic = f'<img src="https://s1.darkroom.com/{pid:028d}" loading="lazy">' if img else ""
    return f'<a class="x" href="{path}">{pic}</a>'


def page(*cards, collections=()):
    return "<html>" + "".join(f'<a href="/collections/{c}">{c}</a>' for c in collections) + "".join(cards) + "</html>"


class FakeSite:
    """Serves `pages` {url: html}; records every request; any other URL is a test failure, not a 404."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url):
        self.calls.append(url)
        if url not in self.pages:
            raise AssertionError(f"unexpected request {url}")
        return self.pages[url]


class Sleeps:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class TestParse:
    def test_both_link_formats_with_their_images_each_product_once(self):
        html = page(card(1350563), card(1933872, "mushrooms"), card(1933872, "mushrooms"))   # the grid repeats a product
        assert darkroom.parse_products(html, BASE + "/") == [
            Product(1350563, None, "https://s1.darkroom.com/0000000000000000000001350563", f"{BASE}/products/1350563"),
            Product(1933872, "mushrooms", "https://s1.darkroom.com/0000000000000000000001933872", f"{BASE}/products/mushrooms/1933872")]

    def test_a_link_without_an_image_is_not_a_product_and_one_with_it_is(self):
        html = page(card(5, img=False), card(6))
        assert [p.product_id for p in darkroom.parse_products(html, BASE)] == [6]


class TestList:
    def site(self):
        return FakeSite({
            f"{BASE}/": page(collections=("insects", "mushrooms")),
            f"{BASE}/?page=1": page(card(10), card(11)),
            f"{BASE}/?page=2": page(card(11)),                                        # nothing new: the listing is over
            f"{BASE}/collections/insects?page=1": page(card(30, "insects")),
            f"{BASE}/collections/insects?page=2": page(),
            f"{BASE}/collections/mushrooms?page=1": page(card(20, "mushrooms"), card(21, "mushrooms")),
            f"{BASE}/collections/mushrooms?page=2": page(card(22, "mushrooms"), card(10, "mushrooms")),   # 10 is already known
            f"{BASE}/collections/mushrooms?page=3": page()})

    def test_home_grid_and_every_collection_are_read_page_by_page_with_a_pause_before_each_request(self):
        site, sleeps = self.site(), Sleeps()
        products = darkroom.list_products(site, base=BASE + "/", sleep=sleeps, pause_s=1.5)
        assert [(p.product_id, p.collection) for p in products] == [(10, None), (11, None), (20, "mushrooms"), (21, "mushrooms"),
                                                                    (22, "mushrooms"), (30, "insects")]
        assert site.calls == [f"{BASE}/", f"{BASE}/?page=1", f"{BASE}/?page=2", f"{BASE}/collections/insects?page=1",
                              f"{BASE}/collections/insects?page=2", f"{BASE}/collections/mushrooms?page=1",
                              f"{BASE}/collections/mushrooms?page=2", f"{BASE}/collections/mushrooms?page=3"]
        assert sleeps.calls == [1.5] * 7

    def test_a_storefront_with_nothing_on_it_is_an_error_and_one_product_is_not(self):
        empty = FakeSite({f"{BASE}/": page(), f"{BASE}/?page=1": page()})
        with pytest.raises(DarkroomError, match=r"^no products found on https://shop\.test \(checked 1 listing\(s\)\): the storefront changed"):
            darkroom.list_products(empty, base=BASE, sleep=Sleeps(), pause_s=0)
        one = FakeSite({f"{BASE}/": page(), f"{BASE}/?page=1": page(card(7)), f"{BASE}/?page=2": page()})
        assert [p.product_id for p in darkroom.list_products(one, base=BASE, sleep=Sleeps(), pause_s=0)] == [7]


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


class TestMatch:
    FILES = {"2024/a.jpg": 100, "2024/b.jpg": 100, "2024/c.jpg": 200}
    RELS = ["2024/a.jpg", "2024/b.jpg", "2024/c.jpg"]

    def match(self, vec, matrix, **kw):
        return darkroom.match_product(9, unit(*vec), self.RELS, np.stack([unit(*r) for r in matrix]), self.FILES,
                                      threshold=kw.get("threshold", 0.97), margin=kw.get("margin", 0.005))

    def test_the_frame_it_looks_like_is_the_match_and_the_closest_other_observation_is_the_rival(self):
        m = self.match((1, 0, 0), [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        assert (m.status, m.obs_id, m.gallery_file, round(m.score, 6), m.rival_obs, round(m.rival_score, 6)) == (
            "match", 100, "2024/a.jpg", 1.0, 200, 0.0)

    def test_below_the_threshold_is_not_a_match_and_the_same_picture_at_the_threshold_is(self):
        far = self.match((1, 0.3, 0), [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        assert (far.status, far.obs_id, round(far.score, 4)) == ("below_threshold", 100, 0.9578)
        assert self.match((1, 0.3, 0), [(1, 0, 0), (0, 1, 0), (0, 0, 1)], threshold=0.95).status == "match"

    def test_a_frame_of_another_observation_within_the_margin_is_ambiguous(self):
        m = self.match((1, 0, 0.001), [(1, 0, 0), (0, 1, 0), (0.0, 0, 1)], margin=2.0)
        assert (m.status, m.obs_id, m.rival_obs) == ("ambiguous", 100, 200)

    def test_a_second_frame_of_the_same_observation_is_not_a_rival(self):
        m = self.match((1, 0, 0), [(1, 0, 0), (1, 0.001, 0), (0, 0, 1)])
        assert (m.status, m.obs_id, m.rival_obs) == ("match", 100, 200)

    def test_a_gallery_of_one_observation_has_no_rival(self):
        files = {"2024/a.jpg": 100, "2024/b.jpg": 100}
        m = darkroom.match_product(1, unit(1, 0), ["2024/a.jpg", "2024/b.jpg"], np.stack([unit(1, 0), unit(0, 1)]), files,
                                   threshold=0.97, margin=0.005)
        assert (m.status, m.rival_obs, m.rival_score) == ("match", None, None)


VECTORS = {b"g1": unit(1, 0, 0), b"g2": unit(0, 1, 0), b"g3": unit(0, 0, 1),
           b"p1": unit(1, 0.01, 0),            # looks like g1 (observation 101)
           b"p1b": unit(1, 0.002, 0),          # looks even more like g1: the better print for observation 101
           b"p2": unit(0, 1, 0.01),            # looks like g2 (observation 102)
           b"pX": unit(1, 1, 1)}               # looks like nothing in the gallery


class Embedder:
    def __init__(self):
        self.calls = []

    def __call__(self, path):
        self.calls.append(path.name)
        return VECTORS[path.read_bytes()]


class Downloads:
    def __init__(self, by_url):
        self.by_url, self.calls = by_url, []

    def __call__(self, url):
        self.calls.append(url)
        return self.by_url[url]


def img(pid):
    return f"https://s1.darkroom.com/{pid:028d}"


class World:
    """A gallery of three files (observations 101, 102, 103), a storefront and the folders the sync reads and writes."""

    def __init__(self, tmp_path, products):
        self.tmp = tmp_path
        self.gallery = tmp_path / "gallery"
        self.gallery.mkdir()
        for name, data in (("g1", b"g1"), ("g2", b"g2"), ("g3", b"g3")):
            (self.gallery / f"2024-{name}.jpg").write_bytes(data)
        self.index = {"files": {"2024-g1.jpg": 101, "2024-g2.jpg": 102, "2024-g3.jpg": 103}}
        self.products = products                 # {product id: (collection or None, image bytes)}
        self.embed = Embedder()
        self.set_site()

    def set_site(self):
        cards = [card(pid, col) for pid, (col, _) in sorted(self.products.items())]
        self.site = FakeSite({f"{BASE}/": page(*cards), f"{BASE}/?page=1": page(*cards), f"{BASE}/?page=2": page()})
        self.dl = Downloads({img(pid): data for pid, (_, data) in self.products.items()})

    def sync(self, **kw):
        self.embed.calls.clear()
        return darkroom.sync(index=self.index, gallery_dir=self.gallery, work_dir=self.tmp / "work", out_file=self.tmp / "prints.csv",
                             manual_file=self.tmp / "manual.csv", base=BASE, embed_file=self.embed, fetch=self.site, download=self.dl,
                             sleep=Sleeps(), pause_s=0, **kw)

    def prints(self):
        return (self.tmp / "prints.csv").read_bytes().decode("utf-8")

    def review(self):
        with open(self.tmp / "work" / "darkroom_review.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class TestSync:
    def test_matching_products_become_print_rows_and_the_rest_go_to_the_review_file(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1"), 2: ("insects", b"p2"), 3: ("insects", b"pX")})
        summary = w.sync()
        assert w.prints() == f"obs_id,url\r\n101,{BASE}/products/1\r\n102,{BASE}/products/insects/2\r\n"
        assert summary == {"products": 3, "downloaded": 3, "gallery_embedded": 3, "gallery_unreadable": 0, "matched_products": 2,
                           "observations": 2, "from_manual": 0, "review": {"below_threshold": 1, "ambiguous": 0}, "written": True}
        assert [(r["product_id"], r["url"], r["status"]) for r in w.review()] == [("3", f"{BASE}/products/insects/3", "below_threshold")]

    def test_a_second_run_fetches_and_embeds_nothing_and_writes_the_same_file(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1"), 2: (None, b"p2")})
        w.sync()
        first = w.prints()
        w.dl.calls.clear()
        summary = w.sync()
        assert (w.dl.calls, w.embed.calls, summary["downloaded"], summary["gallery_embedded"]) == ([], [], 0, 0)
        assert w.prints() == first

    def test_the_same_picture_on_another_image_shard_is_not_fetched_again(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        w.sync()
        for url, html in list(w.site.pages.items()):
            w.site.pages[url] = html.replace("https://s1.darkroom.com/", "https://s2.darkroom.com/")      # the shard host varies per request
        w.dl.calls.clear()
        summary = w.sync()
        assert (w.dl.calls, w.embed.calls, summary["downloaded"]) == ([], [], 0)
        assert w.prints() == f"obs_id,url\r\n101,{BASE}/products/1\r\n"

    def test_a_new_product_is_the_only_one_fetched_and_embedded(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        w.sync()
        w.products[2] = ("mushrooms", b"p2")
        w.set_site()
        summary = w.sync()
        assert (w.dl.calls, w.embed.calls, summary["downloaded"]) == ([img(2)], ["2.jpg"], 1)
        assert w.prints() == f"obs_id,url\r\n101,{BASE}/products/1\r\n102,{BASE}/products/mushrooms/2\r\n"

    def test_a_replaced_product_image_is_fetched_again(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        w.sync()
        w.site.pages[f"{BASE}/"] = w.site.pages[f"{BASE}/?page=1"] = page('<a href="/products/1"><img src="https://s9.darkroom.com/' + "z" * 28 + '"></a>')
        w.dl.by_url["https://s9.darkroom.com/" + "z" * 28] = b"p2"
        w.dl.calls.clear()
        w.sync()
        assert w.dl.calls == ["https://s9.darkroom.com/" + "z" * 28]
        assert w.prints() == f"obs_id,url\r\n102,{BASE}/products/1\r\n"

    def test_one_link_per_observation_the_better_matching_product_wins(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1"), 4: ("general", b"p1b")})
        w.sync()
        assert w.prints() == f"obs_id,url\r\n101,{BASE}/products/general/4\r\n"

    def test_the_owners_manual_rows_win_and_extra_ones_are_added(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        (tmp_path / "manual.csv").write_text("obs_id,url\r\n101,https://print.test/mine\r\n555,https://print.test/other\r\n", encoding="utf-8")
        summary = w.sync()
        assert w.prints() == "obs_id,url\r\n101,https://print.test/mine\r\n555,https://print.test/other\r\n"
        assert (summary["from_manual"], summary["observations"], summary["matched_products"]) == (2, 2, 1)

    def test_a_dry_run_writes_neither_file_and_a_real_run_writes_both(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        assert w.sync(write=False)["written"] is False
        assert not (tmp_path / "prints.csv").exists() and not (tmp_path / "work" / "darkroom_review.csv").exists()
        w.sync()
        assert (tmp_path / "prints.csv").is_file() and (tmp_path / "work" / "darkroom_review.csv").is_file()

    def test_an_unreachable_storefront_leaves_the_existing_prints_file_alone(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        w.sync()
        before = w.prints()
        w.site.pages = {f"{BASE}/": page(), f"{BASE}/?page=1": page()}
        with pytest.raises(DarkroomError, match="no products found on https://shop.test"):
            w.sync()
        assert w.prints() == before

    def test_a_bad_manual_file_is_refused_naming_the_header(self, tmp_path):
        w = World(tmp_path, {1: (None, b"p1")})
        (tmp_path / "manual.csv").write_text("obs,link\r\n1,2\r\n", encoding="utf-8")
        with pytest.raises(DarkroomError, match=r"manual\.csv: needs the header obs_id,url"):
            w.sync()


class TestGalleryVectors:
    def files(self, tmp_path):
        for n in ("a", "b"):
            (tmp_path / f"{n}.jpg").write_bytes(n.encode())
        return {"a.jpg": 1, "b.jpg": 2}

    def embed(self, calls):
        def f(path):
            calls.append(path.name)
            return unit(1, 0) if path.read_bytes() == b"a" else unit(0, 1)
        return f

    def test_cached_vectors_are_reused_and_a_changed_file_is_embedded_again(self, tmp_path):
        files, calls, cache = self.files(tmp_path), [], tmp_path / "cache.npz"
        rels, matrix, bad, fresh = darkroom.gallery_vectors(files, tmp_path, cache, self.embed(calls))
        assert (rels, matrix.tolist(), bad, fresh, calls) == (["a.jpg", "b.jpg"], [[1.0, 0.0], [0.0, 1.0]], [], 2, ["a.jpg", "b.jpg"])
        calls.clear()
        assert darkroom.gallery_vectors(files, tmp_path, cache, self.embed(calls))[3] == 0 and calls == []
        (tmp_path / "b.jpg").write_bytes(b"bb")
        assert darkroom.gallery_vectors(files, tmp_path, cache, self.embed(calls))[3] == 1 and calls == ["b.jpg"]

    def test_an_unreadable_file_is_reported_and_the_rest_is_used(self, tmp_path):
        files = {**self.files(tmp_path), "gone.jpg": 3}
        rels, _, bad, fresh = darkroom.gallery_vectors(files, tmp_path, tmp_path / "cache.npz", self.embed([]))
        assert (rels, bad, fresh) == (["a.jpg", "b.jpg"], ["gone.jpg"], 2)

    def test_a_gallery_where_nothing_can_be_read_is_an_error(self, tmp_path):
        with pytest.raises(DarkroomError, match=r"^no gallery file could be embedded"):
            darkroom.gallery_vectors({"gone.jpg": 3}, tmp_path, tmp_path / "cache.npz", self.embed([]))


class TestCli:
    def test_darkroom_sync_through_the_cli_prints_the_summary(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli, settings

        monkeypatch.setattr(settings, "DARKROOM_DIR", tmp_path / "work")
        monkeypatch.setattr(settings, "SOCIAL_PRINTS_FILE", tmp_path / "prints.csv")
        monkeypatch.setattr(settings, "DARKROOM_URL", BASE)
        monkeypatch.setattr(settings, "DARKROOM_PAUSE_S", 0)
        (tmp_path / "index.json").write_text(json.dumps({"files": {}, "unresolved": {}}), encoding="utf-8")
        monkeypatch.setattr(darkroom, "http_text", FakeSite({f"{BASE}/": page(), f"{BASE}/?page=1": page()}))
        monkeypatch.setattr("sys.argv", ["bio", "darkroom-sync", "--index", str(tmp_path / "index.json"), "--gallery-dir", str(tmp_path)])
        assert cli.main() == 1
        assert capsys.readouterr().err == (f"bio darkroom-sync: no products found on {BASE} (checked 1 listing(s)): "
                                           "the storefront changed or is unreachable\n")

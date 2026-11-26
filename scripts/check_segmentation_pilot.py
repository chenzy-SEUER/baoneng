"""无 GPU/真实数据依赖的分割试验导出检查。"""
import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from scripts.prepare_segmentation_pilot import export, validate_polygons, write_json


class ExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.folder = self.package / "annotations/train"
        self.folder.mkdir(parents=True)
        self.record = {"id": "sample", "split": "train", "width": 20, "height": 20,
                       "objects": [{"label": "zhedang", "box": [1, 1, 12, 12]}]}
        self.annotation = {
            "flags": {"reviewed": False, "draft": True}, "imageWidth": 20,
            "imageHeight": 20, "imagePath": "sample.png",
            "shapes": [{"label": "obj_0", "shape_type": "polygon", "flags": {"draft": True},
                        "points": [[2, 2], [10, 2], [10, 10], [2, 10]]}],
        }
        Image.new("RGB", (20, 20)).save(self.folder / "sample.png")
        self.save()

    def save(self):
        write_json(self.package / "manifest.json", [self.record])
        write_json(self.folder / "sample.json", self.annotation)

    def run_export(self, destination, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            export(self.package, destination, **kwargs)

    def test_drafts_rejected_by_default_without_output(self):
        destination = self.root / "rejected"
        with self.assertRaises(ValueError):
            self.run_export(destination)
        self.assertFalse(destination.exists())

    def test_opt_in_preserves_source_and_paired_geometry(self):
        original = (self.folder / "sample.json").read_bytes()
        destination = self.root / "export"
        self.run_export(destination, allow_drafts=True)
        self.assertEqual(original, (self.folder / "sample.json").read_bytes())
        metadata = json.loads((destination / "experiment.json").read_text())
        self.assertEqual(metadata["label_status"], "experimental_pseudo_labels")
        self.assertFalse(metadata["annotations"][0]["reviewed"])
        self.assertTrue(metadata["annotations"][0]["polygons"][0]["flags"]["draft"])
        det = (destination / "detect/labels/train/sample.txt").read_text().split()
        self.assertEqual(list(map(float, det[1:])), [0.3, 0.3, 0.4, 0.4])
        seg = (destination / "segment/labels/train/sample.txt").read_text().split()
        self.assertEqual(list(map(float, seg[1:])), [.1, .1, .5, .1, .5, .5, .1, .5])
        self.assertEqual((destination / "segment/images/train/sample.png").read_bytes(),
                         (destination / "detect/images/train/sample.png").read_bytes())

    def test_draft_opt_in_still_rejects_bad_polygons(self):
        mutations = []
        missing = deepcopy(self.annotation); missing["shapes"] = []; mutations.append(missing)
        duplicate = deepcopy(self.annotation); duplicate["shapes"] *= 2; mutations.append(duplicate)
        outside = deepcopy(self.annotation); outside["shapes"][0]["points"][0] = [-1, 2]; mutations.append(outside)
        flat = deepcopy(self.annotation); flat["shapes"][0]["points"] = [[1, 1], [2, 2], [3, 3]]; mutations.append(flat)
        for annotation in mutations:
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                validate_polygons(annotation, self.record, allow_drafts=True)

    def test_test_split_and_duplicate_ids_rejected(self):
        self.record["split"] = "test"
        self.save()
        with self.assertRaises(ValueError):
            self.run_export(self.root / "test", allow_drafts=True)
        self.record["split"] = "train"
        write_json(self.package / "manifest.json", [self.record, self.record])
        with self.assertRaises(ValueError):
            self.run_export(self.root / "duplicates", allow_drafts=True)

    def test_existing_destination_not_overwritten(self):
        destination = self.root / "existing"
        destination.mkdir()
        sentinel = destination / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_export(destination, allow_drafts=True)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

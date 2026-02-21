import unittest

import pandas as pd

from main import FeaturePage


class FeatureCLogicTests(unittest.TestCase):
    def setUp(self):
        self.page = FeaturePage.__new__(FeaturePage)

    def test_find_feature_c_columns_with_xlocation_pair(self):
        df = pd.DataFrame(columns=["Net Name", "XLocation", "YLocation", "Etc"])
        net_col, x_col, y_col = self.page._find_feature_c_columns(df)
        self.assertEqual(net_col, "Net Name")
        self.assertEqual(x_col, "XLocation")
        self.assertEqual(y_col, "YLocation")

    def test_find_feature_c_columns_with_xcoord_pair(self):
        df = pd.DataFrame(columns=["Net", "X Coord", "Y Coord"])
        net_col, x_col, y_col = self.page._find_feature_c_columns(df)
        self.assertEqual(net_col, "Net")
        self.assertEqual(x_col, "X Coord")
        self.assertEqual(y_col, "Y Coord")

    def test_build_feature_c_points_filters_invalid_rows(self):
        df = pd.DataFrame(
            {
                "Net Name": ["A", "B", "C"],
                "XLocation": ["1,000.5", "bad", "3.25"],
                "YLocation": ["2.0", "4.0", ""],
            }
        )
        points = self.page._build_feature_c_points(df, "Net Name", "XLocation", "YLocation")
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["net"], "A")
        self.assertEqual(points[0]["x"], 1000.5)
        self.assertEqual(points[0]["y"], 2.0)
        self.assertEqual(points[0]["net_norm"], "a")

    def test_prepare_highlight_nets_preserves_order_and_dedupes(self):
        pasted = ["NET3", "net2", "NET3", "net1"]
        available = {"net1", "net2", "net3"}
        matched, display_map = self.page._prepare_feature_c_highlight_nets(pasted, available)
        self.assertEqual(matched, ["net3", "net2", "net1"])
        self.assertEqual(display_map["net3"], "NET3")

    def test_paginate_items_splits_by_20(self):
        items = [f"net{i}" for i in range(45)]
        pages = self.page._paginate_items(items, 20)
        self.assertEqual(len(pages), 3)
        self.assertEqual(len(pages[0]), 20)
        self.assertEqual(len(pages[1]), 20)
        self.assertEqual(len(pages[2]), 5)

    def test_build_feature_c_color_map_creates_color_for_each_net(self):
        nets = [f"net{i}" for i in range(25)]
        color_map = self.page._build_feature_c_color_map(nets)
        self.assertEqual(len(color_map), 25)
        for net in nets:
            self.assertTrue(color_map[net].startswith("#"))
            self.assertEqual(len(color_map[net]), 7)


if __name__ == "__main__":
    unittest.main()

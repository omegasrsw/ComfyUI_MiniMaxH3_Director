import unittest
import torch

from director.lipsync_refs import normalize_lipsync_refs


class ReferenceTests(unittest.TestCase):
    def test_stable_numeric_order_and_limit(self):
        images = {f'ref_image_{i}': torch.full((1, 32, 32, 3), i / 10) for i in range(9, 0, -1)}
        ordered = normalize_lipsync_refs(images)
        self.assertEqual(list(ordered), [f'ref_image_{i}' for i in range(1, 10)])
        for key in ordered:
            self.assertIs(ordered[key], images[key])

    def test_missing_invalid_and_nonconsecutive_references(self):
        image = torch.zeros(1, 32, 32, 3)
        for refs in ({}, None, {'ref_image_2': image}, {'ref_image_1': image, 'ref_image_3': image},
                     {'ref_image_1': image, 'ref_image_10': image}, {'ref_image_1': image.repeat(2, 1, 1, 1)},
                     {'ref_image_1': torch.full_like(image, float('nan'))}):
            with self.assertRaises(ValueError):
                normalize_lipsync_refs(refs)

    def test_optional_empty_trailing_slots(self):
        image = torch.zeros(1, 32, 32, 3)
        self.assertEqual(list(normalize_lipsync_refs({'ref_image_1': image, 'ref_image_2': None})), ['ref_image_1'])

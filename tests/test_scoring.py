import unittest

from scoring import calculate_confidence


class ConfidenceScoringTests(unittest.TestCase):
    def test_identical_gps_gets_largest_boost(self):
        confidence = calculate_confidence(
            visual_similarity=0.4,
            gps_distance_km=0.0,
            geo_threshold_km=0.2,
            matching_tags=[],
            max_distance_km=0.2,
        )
        self.assertEqual(confidence, 0.9)

    def test_far_gps_distance_zeroes_confidence_when_limit_is_enabled(self):
        confidence = calculate_confidence(
            visual_similarity=0.95,
            gps_distance_km=1.0,
            geo_threshold_km=0.2,
            matching_tags=["pub"],
            max_distance_km=0.2,
        )
        self.assertEqual(confidence, 0.0)

    def test_tags_boost_confidence(self):
        confidence = calculate_confidence(
            visual_similarity=0.7,
            gps_distance_km=None,
            geo_threshold_km=0.2,
            matching_tags=["stone", "church", "tower"],
            max_distance_km=None,
        )
        self.assertAlmostEqual(confidence, 0.85)


if __name__ == "__main__":
    unittest.main()

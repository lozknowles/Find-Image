EPSILON = 1e-6


def calculate_confidence(
    visual_similarity: float,
    gps_distance_km: float | None,
    geo_threshold_km: float,
    matching_tags: list[str],
    max_distance_km: float | None,
) -> float:
    if max_distance_km is not None and gps_distance_km is not None and gps_distance_km > max_distance_km:
        return 0.0

    confidence = visual_similarity

    tag_count = len(matching_tags)
    if tag_count == 1:
        confidence += 0.075
    elif tag_count == 2:
        confidence += 0.1
    elif tag_count > 2:
        confidence += 0.1 + (tag_count - 2) * 0.05

    if gps_distance_km is not None:
        if gps_distance_km < EPSILON:
            confidence += 0.5
        elif gps_distance_km < 0.05:
            confidence += 0.1
        elif gps_distance_km <= (geo_threshold_km + EPSILON):
            confidence += 0.1

    return max(min(confidence, 1.0), 0.0)

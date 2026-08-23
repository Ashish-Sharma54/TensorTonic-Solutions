def generate_anchors(
    feature_size: int,
    image_size: float,
    scales: list[float],
    aspect_ratios: list[float]
) -> list[list[float]]:
    """
    Generate anchor boxes for object detection.
    """

    stride = image_size / feature_size

    anchors = []

    for i in range(feature_size):
        for j in range(feature_size):

            # Center of grid cell
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for s in scales:
                for r in aspect_ratios:

                    # Width and height
                    w = s * (r ** 0.5)
                    h = s / (r ** 0.5)

                    # Convert center-width-height to corners
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ])

    return anchors
"""Direct arrow relationships, grounded in rectangular nodes and source OCR."""

from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag.application.answer import ModelProtocolError
from enterprise_rag.application.converted_visual_answer import exact_source_label
from enterprise_rag.application.visual_regions import VisualRegionService
from enterprise_rag.ingestion.ocr import OCRTextRegion
from enterprise_rag.models.raster_geometry import _read_scene, _rectangle_nodes

BBox = tuple[float, float, float, float]
MAX_PATH_HOPS = 8


@dataclass(frozen=True)
class DiagramNode:
    label: str
    bbox: BBox


@dataclass(frozen=True)
class DiagramEdge:
    source: int
    target: int
    bbox: BBox
    condition: str | None = None
    condition_bbox: BBox | None = None


@dataclass(frozen=True)
class DiagramGraph:
    nodes: tuple[DiagramNode, ...]
    edges: tuple[DiagramEdge, ...]
    complete: bool


@dataclass(frozen=True)
class StepQuestion:
    direction: str
    label: str
    hops: int | None = 1
    condition: str | None = None


def parse_step_question(question: str) -> StepQuestion | None:
    conditional = re.fullmatch(
        r"\s*(?:(?:which|what) (?:step|node) follows|какой (?:шаг|блок) следует после) (.+?) (?:when|if|при условии) ([^\W\d_]{1,40})\s*[?.!]?\s*",
        question,
        re.I,
    )
    if conditional and 1 <= len(conditional[1].strip()) <= 200:
        return StepQuestion("out", conditional[1].strip(), 1, conditional[2])
    counted = (
        r"(?:which|what) (?:step|node) is(?: reached)? (\d{1,2}) (?:steps?|transitions?) (after|before) (.+?)",
        r"какой (?:шаг|блок) (?:будет|находится) через (\d{1,2}) (?:шага|шагов|перехода|переходов) (после|перед) (.+?)",
    )
    for pattern in counted:
        match = re.fullmatch(r"\s*" + pattern + r"\s*[?.!]?\s*", question, re.I)
        if match:
            hops = int(match[1])
            label = match[3].strip()
            if 1 <= hops <= MAX_PATH_HOPS and 1 <= len(label) <= 200:
                return StepQuestion(
                    "out" if match[2].casefold() in {"after", "после"} else "in",
                    label,
                    hops,
                )
            return None
    terminal = re.fullmatch(
        r"\s*(?:(?:which|what) (?:final|last) (?:step|node) is reached from|какой (?:конечный|последний) (?:шаг|блок) достигается из) (.+?)\s*[?.!]?\s*",
        question,
        re.I,
    )
    if terminal and 1 <= len(terminal[1].strip()) <= 200:
        return StepQuestion("out", terminal[1].strip(), None)
    patterns = (
        ("out", r"(?:which|what) (?:step|label|node) (?:follows|comes after) (.+?)"),
        ("in", r"(?:which|what) (?:step|label|node) (?:precedes|comes before) (.+?)"),
        ("out", r"какой (?:шаг|блок) (?:следует|идёт|идет) после (.+?)"),
        ("in", r"какой (?:шаг|блок) (?:находится|идёт|идет) перед (.+?)"),
    )
    for direction, pattern in patterns:
        match = re.fullmatch(r"\s*" + pattern + r"\s*[?.!]?\s*", question, re.I)
        if match and 1 <= len(match[1].strip()) <= 200:
            return StepQuestion(direction, match[1].strip())
    return None


def graph_from_payload(value: object) -> DiagramGraph:
    try:
        if (
            not isinstance(value, dict)
            or set(value) != {"nodes", "edges", "complete"}
            or type(value["complete"]) is not bool
        ):
            raise ValueError
        if (
            not isinstance(value["nodes"], list)
            or not isinstance(value["edges"], list)
            or len(value["nodes"]) > 32
            or len(value["edges"]) > 64
        ):
            raise ValueError
        nodes = []
        for node in value["nodes"]:
            if (
                not isinstance(node, dict)
                or set(node) != {"label", "bbox"}
                or not isinstance(node["label"], str)
                or not 1 <= len(node["label"]) <= 600
            ):
                raise ValueError
            box = VisualRegionService._bbox(node["bbox"])
            if box is None:
                raise ValueError
            nodes.append(DiagramNode(node["label"], box))
        if len({n.label for n in nodes}) != len(nodes):
            raise ValueError
        edges = []
        seen = set()
        for edge in value["edges"]:
            if not isinstance(edge, dict) or set(edge) not in (
                {"source", "target", "bbox"},
                {"source", "target", "bbox", "condition", "condition_bbox"},
            ):
                raise ValueError
            start, end = edge["source"], edge["target"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < len(nodes)
                or not 0 <= end < len(nodes)
                or start == end
                or (start, end) in seen
            ):
                raise ValueError
            box = VisualRegionService._bbox(edge["bbox"])
            if box is None or any(
                not (
                    box[0] <= node.bbox[0] < node.bbox[2] <= box[2]
                    and box[1] <= node.bbox[1] < node.bbox[3] <= box[3]
                )
                for node in (nodes[start], nodes[end])
            ):
                raise ValueError
            seen.add((start, end))
            condition, condition_bbox = (
                edge.get("condition"),
                edge.get("condition_bbox"),
            )
            if condition is not None:
                if (
                    not isinstance(condition, str)
                    or not condition.isalpha()
                    or not 1 <= len(condition) <= 40
                ):
                    raise ValueError
                condition_bbox = VisualRegionService._bbox(condition_bbox)
                if condition_bbox is None or not (
                    box[0] <= condition_bbox[0] < condition_bbox[2] <= box[2]
                    and box[1] <= condition_bbox[1] < condition_bbox[3] <= box[3]
                ):
                    raise ValueError
            elif condition_bbox is not None:
                raise ValueError
            edges.append(DiagramEdge(start, end, box, condition, condition_bbox))
        return DiagramGraph(tuple(nodes), tuple(edges), value["complete"])
    except (ValueError, TypeError, KeyError):
        raise ModelProtocolError("Invalid source-bound diagram graph") from None


def select_step(graph: DiagramGraph, question: StepQuestion, source: str):
    if question.condition is not None and (
        question.direction != "out" or question.hops != 1
    ):
        return None
    if question.direction not in {"out", "in"} or (
        question.hops is not None
        and (type(question.hops) is not int or not 1 <= question.hops <= MAX_PATH_HOPS)
    ):
        return None
    if not graph.complete or any(
        exact_source_label(source, node.label) is None for node in graph.nodes
    ):
        return None
    quote = exact_source_label(source, question.label)
    matches = [i for i, node in enumerate(graph.nodes) if node.label == quote]
    if len(matches) != 1:
        return None
    index = matches[0]
    seen = {index}
    path: list[DiagramEdge] = []
    terminal = question.hops is None
    limit = len(graph.nodes) - 1 if terminal else question.hops
    assert limit is not None
    for _ in range(limit):
        edges = [
            edge
            for edge in graph.edges
            if (edge.source if question.direction == "out" else edge.target) == index
        ]
        if (
            question.condition is None
            and question.direction == "out"
            and any(edge.condition is not None for edge in edges)
        ):
            return None
        if question.condition is not None:
            if any(
                edge.condition is None or edge.condition not in source for edge in edges
            ) or len(
                {edge.condition.casefold() for edge in edges if edge.condition}
            ) != len(edges):
                return None
            edges = [
                edge
                for edge in edges
                if edge.condition is not None
                and edge.condition.casefold() == question.condition.casefold()
            ]
        if not edges and terminal and path:
            break
        if len(edges) != 1:
            return None
        edge = edges[0]
        index = edge.target if question.direction == "out" else edge.source
        if index in seen:
            return None
        seen.add(index)
        path.append(edge)
    if not path or (terminal and any(edge.source == index for edge in graph.edges)):
        return None
    boxes = [edge.bbox for edge in path]
    bbox = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    return graph.nodes[index].label, bbox


def extract_graph(
    png: bytes, source: str, regions: tuple[OCRTextRegion, ...]
) -> DiagramGraph:
    scene = _read_scene(png, source, regions)
    cv, np, ink, width, height = (
        scene.cv,
        scene.np,
        scene.ink.copy(),
        scene.width,
        scene.height,
    )
    rectangles = _rectangle_nodes(scene)
    if not 2 <= len(rectangles) <= 32:
        return DiagramGraph((), (), False)
    nodes = tuple(DiagramNode(label, scene.norm(box)) for box, label in rectangles)
    if len({node.label for node in nodes}) != len(nodes):
        return DiagramGraph((), (), False)
    pad = max(2, round(min(width, height) * 0.006))
    for box, _ in rectangles:
        x, y, right, bottom = box
        ink[
            max(0, y - pad) : min(height, bottom + pad),
            max(0, x - pad) : min(width, right + pad),
        ] = 0
    for region in regions:
        # OCR may misread a drawn arrow as currency/punctuation. Such a region
        # must not erase the very arrow that provides the relationship evidence.
        if region.confidence >= 0.5 and any(char.isalnum() for char in region.text):
            x, y, right, bottom = region.bbox
            ink[
                max(0, round(y * height) - pad) : min(
                    height, round(bottom * height) + pad
                ),
                max(0, round(x * width) - pad) : min(width, round(right * width) + pad),
            ] = 0
    count, components, stats, _ = cv.connectedComponentsWithStats(ink, 8)
    if count > 8192:
        raise ValueError("Connector component limit exceeded")
    edges = []
    connector_boxes = []
    complete = True
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if max(w, h) < min(width, height) * 0.025 or area < 12:
            continue
        horizontal = w >= 2.5 * h
        vertical = h >= 2.5 * w
        if not horizontal and not vertical:
            complete = False
            continue
        mask = components[y : y + h, x : x + w] == index
        profile = np.count_nonzero(mask, axis=0 if horizontal else 1)
        length = len(profile)
        peak = int(profile.max())
        median = float(np.median(profile))
        wide = np.flatnonzero(profile >= peak * 0.65)
        if (
            median <= 0
            or peak < median * 1.8
            or peak < max(5, min(width, height) * 0.009)
            or len(wide) < 2
        ):
            complete = False
            continue
        head_at_end = wide[0] > 0.6 * length
        head_at_start = wide[-1] < 0.4 * length
        if head_at_end == head_at_start:
            complete = False
            continue
        facing = profile if head_at_end else profile[::-1]
        offsets = np.arange(h if horizontal else w)
        centers = (mask * (offsets[:, None] if horizontal else offsets[None, :])).sum(
            axis=0 if horizontal else 1
        ) / profile
        shaft = (centers if head_at_end else centers[::-1])[: round(length * 0.6)]
        if float(shaft.max() - shaft.min()) > max(2, min(width, height) * 0.006):
            complete = False
            continue
        if (
            facing[-1] > 0.55 * peak
            or np.count_nonzero(facing[: round(length * 0.6)] <= 0.55 * peak)
            < length * 0.5
        ):
            complete = False
            continue
        low = (x, y + h / 2) if horizontal else (x + w / 2, y)
        high = (x + w, y + h / 2) if horizontal else (x + w / 2, y + h)
        tail, tip = (low, high) if head_at_end else (high, low)
        direction = (1 if head_at_end else -1) * (1 if horizontal else 2)

        def endpoints(point, is_target, direction=direction):
            matches = []
            for i, (box, _) in enumerate(rectangles):
                left, top, right, bottom = box
                if abs(direction) == 1:
                    face = left if (direction > 0) == is_target else right
                    gap = (
                        (face - point[0])
                        * (1 if direction > 0 else -1)
                        * (1 if is_target else -1)
                    )
                    aligned = top + pad < point[1] < bottom - pad
                else:
                    face = top if (direction > 0) == is_target else bottom
                    gap = (
                        (face - point[1])
                        * (1 if direction > 0 else -1)
                        * (1 if is_target else -1)
                    )
                    aligned = left + pad < point[0] < right - pad
                if aligned and -pad <= gap <= min(width, height) * 0.04:
                    matches.append(i)
            return matches

        starts, ends = endpoints(tail, False), endpoints(tip, True)
        if len(starts) != 1 or len(ends) != 1 or starts[0] == ends[0]:
            complete = False
            continue
        start, end = starts[0], ends[0]
        boxes = [rectangles[start][0], rectangles[end][0], (x, y, x + w, y + h)]
        bbox = scene.norm(
            (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
        )
        edges.append(DiagramEdge(start, end, bbox))
        connector_boxes.append((x, y, w, h, horizontal))
    if len(edges) > 64 or len({(e.source, e.target) for e in edges}) != len(edges):
        complete = False
    assignments: dict[int, list[OCRTextRegion]] = {}
    for region in regions:
        if (
            region.confidence < 0.5
            or not region.text.isalpha()
            or len(region.text) > 40
        ):
            continue
        if any(
            region.bbox[0] < node.bbox[2]
            and region.bbox[2] > node.bbox[0]
            and region.bbox[1] < node.bbox[3]
            and region.bbox[3] > node.bbox[1]
            for node in nodes
        ):
            continue
        cx = (region.bbox[0] + region.bbox[2]) * width / 2
        cy = (region.bbox[1] + region.bbox[3]) * height / 2
        matches = []
        for i, (x, y, w, h, horizontal) in enumerate(connector_boxes):
            along = (cx - x) / w if horizontal else (cy - y) / h
            across = abs(cy - (y + h / 2)) if horizontal else abs(cx - (x + w / 2))
            if 0.2 <= along <= 0.8 and pad < across <= min(width, height) * 0.075:
                matches.append(i)
        if len(matches) == 1:
            assignments.setdefault(matches[0], []).append(region)
    tagged = []
    for i, edge in enumerate(edges):
        labels = assignments.get(i, [])
        if len(labels) == 1:
            region = labels[0]
            boxes = [edge.bbox, region.bbox]
            box = (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            tagged.append(
                DiagramEdge(edge.source, edge.target, box, region.text, region.bbox)
            )
        else:
            tagged.append(edge)
    return DiagramGraph(nodes, tuple(tagged), complete)

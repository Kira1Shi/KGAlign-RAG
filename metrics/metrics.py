from collections import defaultdict, deque


def norm(value):
    return " ".join(str(value).casefold().replace("_", " ").split())


# Relation matcher is used to work with synonimical expressions as if
# they are identical
def edge_matches(left, right, relation_matcher=None):
    if relation_matcher is None:
        relation_matcher = lambda a, b: a == b

    return (
        left[0] == right[0]
        and left[2] == right[2]
        and relation_matcher(left[1], right[1])
    )


def unsupported_entities(answer_triples, context_triples):
    answer_nodes = set()
    for triple in answer_triples:
        answer_nodes.add(norm(triple["subject"]))
        answer_nodes.add(norm(triple["object"]))

    context_nodes = set()
    for triple in context_triples:
        context_nodes.add(norm(triple["subject"]))
        context_nodes.add(norm(triple["object"]))

    return answer_nodes - context_nodes


def unsupported_edges(answer_triples, context_triples, relation_matcher=None):
    answer_edges = {
        (norm(t["subject"]), norm(t["relation"]), norm(t["object"]))
        for t in answer_triples
    }
    context_edges = {
        (norm(t["subject"]), norm(t["relation"]), norm(t["object"]))
        for t in context_triples
    }

    return {
        e for e in answer_edges
        if not any(edge_matches(e, c, relation_matcher) for c in context_edges)
    }


def query_relevant_unsupported_edges(
    answer_triples,
    context_triples,
    query_triples,
    relation_matcher=None,
):
    query_nodes = set()
    query_relations = set()
    for triple in query_triples:
        query_nodes.add(norm(triple["subject"]))
        query_nodes.add(norm(triple["object"]))
        query_relations.add(norm(triple["relation"]))

    return {
        e for e in unsupported_edges(answer_triples, context_triples, relation_matcher)
        if e[0] in query_nodes or e[2] in query_nodes or e[1] in query_relations
    }


def entity_grounding(answer_triples, context_triples):
    answer_triples = list(answer_triples)
    context_triples = list(context_triples)

    answer_nodes = set()
    for triple in answer_triples:
        answer_nodes.add(norm(triple["subject"]))
        answer_nodes.add(norm(triple["object"]))
    if not answer_nodes:
        return None

    return 1.0 - len(unsupported_entities(answer_triples, context_triples)) / len(answer_nodes)


def relation_preservation(
    answer_triples,
    context_triples,
    relation_matcher=None,
):
    answer_triples = list(answer_triples)
    context_triples = list(context_triples)

    answer_edges = {
        (norm(t["subject"]), norm(t["relation"]), norm(t["object"]))
        for t in answer_triples
    }
    if not answer_edges:
        return None

    unsupported = unsupported_edges(answer_triples, context_triples, relation_matcher)
    return 1.0 - len(unsupported) / len(answer_edges)


def weighted_entity_grounding(
    answer_triples,
    context_triples,
    query_triples=(),
    query_boost=2.0,
):
    answer_triples = list(answer_triples)
    context_triples = list(context_triples)
    query_triples = list(query_triples)

    query_nodes = set()
    for triple in query_triples:
        query_nodes.add(norm(triple["subject"]))
        query_nodes.add(norm(triple["object"]))

    answer_nodes = set()
    for triple in answer_triples:
        answer_nodes.add(norm(triple["subject"]))
        answer_nodes.add(norm(triple["object"]))
    if not answer_nodes:
        return None

    def weight(node):
        return 1.0 + query_boost * (node in query_nodes)

    unsupported = unsupported_entities(answer_triples, context_triples)
    total = sum(weight(n) for n in answer_nodes)
    grounded = sum(
        weight(n)
        for n in answer_nodes - unsupported
    )
    return grounded / total


def weighted_relation_preservation(
    answer_triples,
    context_triples,
    query_triples=(),
    endpoint_boost=1.0,
    relation_boost=0.0,
    relation_matcher=None,
):
    answer_triples = list(answer_triples)
    context_triples = list(context_triples)
    query_triples = list(query_triples)

    query_nodes = set()
    query_relations = set()
    for triple in query_triples:
        query_nodes.add(norm(triple["subject"]))
        query_nodes.add(norm(triple["object"]))
        query_relations.add(norm(triple["relation"]))

    answer_edges = {
        (norm(t["subject"]), norm(t["relation"]), norm(t["object"]))
        for t in answer_triples
    }
    if not answer_edges:
        return None

    def weight(e):
        subject, relation, object_ = e
        return (
            1.0
            + endpoint_boost * (subject in query_nodes)
            + endpoint_boost * (object_ in query_nodes)
            + relation_boost * (relation in query_relations)
        )

    unsupported = unsupported_edges(answer_triples, context_triples, relation_matcher)
    total = sum(weight(e) for e in answer_edges)
    supported = sum(
        weight(e)
        for e in answer_edges - unsupported
    )
    return supported / total


def subgraph_connectivity(answer_triples, context_triples):
    answer_triples = list(answer_triples)
    context_triples = list(context_triples)

    answer_edges = {
        (norm(t["subject"]), norm(t["relation"]), norm(t["object"]))
        for t in answer_triples
    }
    if not answer_edges:
        return None

    supported = answer_edges - unsupported_edges(answer_triples, context_triples)
    if not supported:
        return 0.0

    return largest_component_size(supported) / len(answer_edges)


def query_coverage(query_triples, answer_triples, relation_matcher=None):
    query_triples = list(query_triples)
    query_nodes = set()
    query_edges = set()
    for triple in query_triples:
        query_nodes.add(norm(triple["subject"]))
        query_nodes.add(norm(triple["object"]))
        query_edges.add((
            norm(triple["subject"]),
            norm(triple["relation"]),
            norm(triple["object"]),
        ))
    if not query_nodes and not query_edges:
        return None

    answer_nodes = set()
    answer_edges = set()
    for triple in answer_triples:
        answer_nodes.add(norm(triple["subject"]))
        answer_nodes.add(norm(triple["object"]))
        answer_edges.add((
            norm(triple["subject"]),
            norm(triple["relation"]),
            norm(triple["object"]),
        ))
    scores = []

    if query_nodes:
        scores.append(len(query_nodes & answer_nodes) / len(query_nodes))

    if query_edges:
        covered = [
            e for e in query_edges
            if any(edge_matches(e, a, relation_matcher) for a in answer_edges)
        ]
        scores.append(len(covered) / len(query_edges))

    return sum(scores) / len(scores)


def path_support(
    answer_triples,
    context_triples,
    max_depth=None,
    directed=True,
):
    answer_edges = {
        (norm(t["subject"]), norm(t["relation"]), norm(t["object"]))
        for t in answer_triples
    }
    if not answer_edges:
        return None

    graph = defaultdict(set)
    for triple in context_triples:
        subject = norm(triple["subject"])
        object_ = norm(triple["object"])
        graph[subject].add(object_)
        if not directed:
            graph[object_].add(subject)

    supported = sum(
        has_path(graph, subject, object_, max_depth)
        for subject, _, object_ in answer_edges
    )
    return supported / len(answer_edges)


def calculate_metrics(
    answer_triples,
    context_triples,
    query_triples=(),
    entity_query_boost=2.0,
    edge_endpoint_boost=1.0,
    edge_relation_boost=0.0,
):
    answer = list(answer_triples)
    context = list(context_triples)
    query = list(query_triples)

    return {
        "EG": entity_grounding(answer, context, query),
        "RP": relation_preservation(answer, context, query),
        "SC": subgraph_connectivity(answer, context, query),
        "query_coverage": query_coverage(query, answer),
        "path_support": path_support(answer, context, query),
        "weighted_EG": weighted_entity_grounding(
            answer, context, query, entity_query_boost
        ),
        "weighted_RP": weighted_relation_preservation(
            answer, context, query, edge_endpoint_boost, edge_relation_boost
        ),
    }


def linear_combination(metrics, coefficients, intercept=0.0, normalize_weights=False):
    used = {
        name: value
        for name, value in metrics.items()
        if name in coefficients and value is not None
    }
    if not used:
        return None

    score = intercept + sum(coefficients[name] * value for name, value in used.items())
    if normalize_weights:
        weight_sum = sum(abs(coefficients[name]) for name in used)
        score = intercept + (score - intercept) / weight_sum if weight_sum else score

    return score


def has_path(graph, start, goal, max_depth=None):
    if start == goal:
        return True

    queue = deque([(start, 0)])
    seen = {start}

    while queue:
        node, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue

        for neighbor in graph[node]:
            if neighbor == goal:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))

    return False


def largest_component_size(edges):
    graph = defaultdict(set)
    for e in edges:
        subject, _, object_ = e
        graph[subject].add((object_, e))
        graph[object_].add((subject, e))

    best = 0
    seen = set()
    for start in graph:
        if start in seen:
            continue

        queue = deque([start])
        component_edges = set()
        seen.add(start)

        while queue:
            node = queue.popleft()

            for neighbor, e in graph[node]:
                component_edges.add(e)
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        best = max(best, len(component_edges))

    return best

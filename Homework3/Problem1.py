import sys
import math
from collections import Counter, defaultdict


# -----------------------------
# I/O helpers
# -----------------------------
def read_dataset(path: str) -> tuple[list[str], list[list[str]]]:
    """
    Reads a dataset where class label is the first column.
    Works with comma-separated or whitespace-separated values.
    Returns (header, rows). If no header is present in the file, a synthetic header is created.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [
            ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")
        ]

    # Detect delimiter: comma if any comma in first non-empty, else split on whitespace
    if not lines:
        raise ValueError(f"No data in {path}")

    sample = lines[0]
    if "," in sample:
        rows = [ln.split(",") for ln in lines]
    else:
        rows = [ln.split() for ln in lines]

    # Define header(attribute) names
    header = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]
    return header, rows


# -----------------------------
# Entropy & Information Gain
# -----------------------------
def entropy(labels: list[str]) -> float:
    """Shannon entropy H(Y) in bits."""
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def conditional_entropy(subsets: list[list[str]], totals: list[int]) -> float:
    """H(Y|X) = sum_v P(X=v) * H(Y | X=v)."""
    total = sum(totals)
    if total == 0:
        return 0.0
    # conditional entropy
    ce = 0.0
    for i in range(len(subsets)):
        labels_v = subsets[i]
        size_v = totals[i]
        if size_v == 0:
            continue
        ce += (size_v / total) * entropy(labels_v)
    return ce


def information_gain(
    dataset: list[list[str]], attribute_index: int, class_index: int = 0
) -> float:
    """IG(Y; X_attr) for a categorical attribute."""
    # Parent entropy
    parent_labels = [row[class_index] for row in dataset]
    H_parent = entropy(parent_labels)

    # Partition by attribute value
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for row in dataset:
        groups[row[attribute_index]].append(row)

    # Collect labels per value
    subsets_labels = [[r[class_index] for r in grp] for grp in groups.values()]
    totals = [len(grp) for grp in groups.values()]

    H_cond = conditional_entropy(subsets_labels, totals)
    return H_parent - H_cond


# -----------------------------
# Decision Tree
# -----------------------------
class Node:
    def __init__(
        self,
        is_leaf: bool = False,
        prediction: str | None = None,
        split_attr: int | None = None,
        ig_at_node: float = 0.0,
    ):
        self.is_leaf = is_leaf
        self.prediction = prediction  # for leaves
        self.split_attr = split_attr  # attribute index (int) for internal nodes
        self.children: dict[str, "Node"] = {}  # value -> child
        self.ig_at_node = ig_at_node  # max IG used at this split (0 for leaf)

    def pretty(
        self,
        header: list[str],
        prefix: str = "",
        is_last: bool = True,
        edge_label: str | None = None,
    ) -> list[str]:
        """Pretty-print the decision tree like a directory structure."""
        lines = []
        connector = "|__" if is_last else "|--"
        edge = f"{edge_label} -> " if edge_label else ""

        if self.is_leaf:
            lines.append(f"{prefix}{connector}{edge}LEAF: class={self.prediction}")
        else:
            attr_name = header[self.split_attr]
            lines.append(
                f"{prefix}{connector}{edge}split: {attr_name} (IG={self.ig_at_node:.4f})"
            )
            child_prefix = prefix + ("    " if is_last else "|  ")
            # Sort children alphabetically for consistency
            keys = sorted(self.children.keys())
            for i, val in enumerate(keys):
                child_is_last = i == len(keys) - 1
                child_lines = self.children[val].pretty(
                    header, child_prefix, child_is_last, edge_label=val
                )
                lines.extend(child_lines)
        return lines


def majority_class(rows: list[list[str]], class_index: int = 0) -> str:
    counts = Counter(row[class_index] for row in rows)
    # Break ties deterministically by lexicographic class label
    max_count = max(counts.values()) if counts else 0
    candidates = sorted([c for c, k in counts.items() if k == max_count])
    return candidates[0] if candidates else ""


def all_same_class(rows: list[list[str]], class_index: int = 0) -> bool:
    if not rows:
        return True
    first = rows[0][class_index]
    return all(row[class_index] == first for row in rows)


def train(
    rows: list[list[str]],
    header: list[str],
    class_index: int = 0,
    candidate_attrs: list[int] | None = None,
    report: list[tuple[int, float]] | None = None,
) -> Node:
    """
    Trains a decision tree.
    - rows: dataset
    - header: column names (index 0 is 'class')
    - class_index: index of class label (0)
    - candidate_attrs: indices of attributes eligible for splitting (defaults to all except class)
    - report: accumulates (split_attr_index, max_IG) for each internal node (preorder)
    """
    if candidate_attrs is None:
        candidate_attrs = [i for i in range(len(header)) if i != class_index]
    if report is None:
        report = []

    # Base cases
    if not rows:
        # Empty dataset: return a leaf with empty prediction (should not happen in well-formed recursion)
        return Node(is_leaf=True, prediction="", ig_at_node=0.0)
    if all_same_class(rows, class_index):
        return Node(is_leaf=True, prediction=rows[0][class_index], ig_at_node=0.0)
    if not candidate_attrs:
        # No attributes left: predict majority
        return Node(
            is_leaf=True, prediction=majority_class(rows, class_index), ig_at_node=0.0
        )

    # Compute IG for each candidate attribute
    igs = []
    for a in candidate_attrs:
        ig = information_gain(rows, a, class_index)
        igs.append((a, ig))

    # Pick attribute with highest IG; break ties by smaller index (i.e., earlier column, left-to-right)
    max_ig = max(ig for _, ig in igs)
    # All attrs with max IG
    max_attrs = [a for a, ig in igs if abs(ig - max_ig) < 1e-12]
    # Tie-break rule: earliest attribute index wins
    best_attr = min(max_attrs)

    # Record for reporting
    report.append((best_attr, max_ig))

    # Split data by best_attr values
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        groups[row[best_attr]].append(row)

    # Create internal node
    node = Node(is_leaf=False, split_attr=best_attr, ig_at_node=max_ig)

    # Recurse for each attribute value
    new_candidates = [i for i in candidate_attrs if i != best_attr]
    for val, subset in groups.items():
        if not subset:
            child = Node(
                is_leaf=True,
                prediction=majority_class(rows, class_index),
                ig_at_node=0.0,
            )
        else:
            child = train(subset, header, class_index, new_candidates, report)
        node.children[val] = child

    return node


# -----------------------------
# Prediction & Evaluation
# -----------------------------
def predict_one(
    node: Node, row: list[str], default_class: str, class_index: int = 0
) -> str:
    cur = node
    while not cur.is_leaf:
        a = cur.split_attr
        if a is None:
            return default_class
        val = row[a]
        if val in cur.children:
            cur = cur.children[val]
        else:
            # Unseen value at test time: back off to majority class at training root
            return default_class
    return cur.prediction if cur.prediction != "" else default_class


def accuracy(
    node: Node, rows: list[list[str]], default_class: str, class_index: int = 0
) -> float:
    if not rows:
        return 0.0
    # counter for correct predictions
    correct = 0
    for r in rows:
        if predict_one(node, r, default_class, class_index) == r[class_index]:
            correct += 1
    return correct / len(rows)


# -----------------------------
# Main routine
# -----------------------------
def main():

    train_path = "mush_train.data"
    test_path = "mush_test.data"

    header_train, train_rows = read_dataset(train_path)
    header_test, test_rows = read_dataset(test_path)

    # Sanity check: column counts must match
    if len(train_rows[0]) != len(test_rows[0]):
        raise ValueError("Train and test have different number of columns.")

    # Build tree
    report: list[tuple[int, float]] = []
    tree = train(
        train_rows, header_train, class_index=0, candidate_attrs=None, report=report
    )

    # Print tree with IG
    print("\n=== Decision Tree ===")
    for line in tree.pretty(header_train):
        print(line)

    # Report max information gain per node in preorder (split order)
    print("\n=== Split report (node order) ===")
    for idx, (attr_idx, ig) in enumerate(report, 1):
        print(f"Node {idx}: split on '{header_train[attr_idx]}' with max IG = {ig:.5f}")

    # Evaluate on test
    default_cls = majority_class(train_rows, class_index=0)  # used for backoff
    acc = accuracy(tree, test_rows, default_cls, class_index=0)
    print(f"\n=== Test Accuracy ===")
    print(f"Accuracy on {len(test_rows)} examples: {acc*100:.2f}%")


if __name__ == "__main__":
    main()

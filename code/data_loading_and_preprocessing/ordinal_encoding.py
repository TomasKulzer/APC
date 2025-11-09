import numpy as np

class OrdinalEncoder:
    def __init__(self, class_order):
        """
        class_order: List of class names in desired ordinal order.
        Example: ["resistor", "capacitor", "transistor", "IC"]
        """
        self.class_order = class_order
        self.class_to_index = {name: idx for idx, name in enumerate(class_order)}
        self.num_classes = len(class_order)

    def encode(self, labels):
        """
        labels: list or array of class labels (ints or class names)
        Returns: ordinal multi-labels as numpy array (n_samples, num_classes-1)
        """
        # If labels are class names, map them to indices
        if isinstance(labels[0], str):
            labels = [self.class_to_index[label] for label in labels]
        ordinal_labels = np.zeros((len(labels), self.num_classes - 1), dtype=int)
        for i, label in enumerate(labels):
            ordinal_labels[i, :label] = 1
        return ordinal_labels

    def decode(self, ordinal_labels):
        """
        Converts ordinal multi-label back to single-class label (int).
        Returns a list of class indices.
        """
        return [np.sum(row) for row in ordinal_labels]

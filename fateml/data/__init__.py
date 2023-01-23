import bike_rental
import cervical_cancer
import diabetes
import iris


_DATASETS = {
    "iris": iris.load_dataset,
    "diabetes": diabetes.load_dataset,
    "cervical_cancer": cervical_cancer.load_dataset,
    "bike_rental": bike_rental.load_dataset
}


def load_dataset(name: str, **kwargs):
    try:
        return _DATASETS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Given dataset name is not available: {name}")

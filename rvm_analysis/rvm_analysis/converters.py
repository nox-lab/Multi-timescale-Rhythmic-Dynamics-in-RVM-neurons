import pandas as pd

from typing import Iterable

def names_from_classes(classes: Iterable[object]) -> list[str]:
    """Generates a cell name from the matlab labelling scheme used by the Heinricher lab."""
    number_name_dict = {
        0: "Unknown",
        1: "ON",
        2: "OFF",
        3: "Neutral",
        4: "Ignore",
    }

    names: list[str] = []

    for x in classes:
        try:
            label = int(x)  # type: ignore
        except (TypeError, ValueError):
            names.append("Unlabelled")
        else:
            names.append(number_name_dict.get(label, "Unlabelled"))

    return names


def combine_neutral_and_save(file_obj, output_path="combined_data.csv"):
    """Combines the NEUTRAL_EXTRA with the NEUTRAL cells and saves the result to a dataframe."""
    # Load CSV
    df = pd.read_csv(file_obj)

    # Combine NEUTRAL and NEUTRAL_EXTRA into a unified 'NEUTRAL' category
    df['cell_name'] = df['cell_name'].replace({'NEUTRAL_EXTRA': 'NEUTRAL'})

    # Save to new CSV
    df.to_csv(output_path, index=False)

    print(f"Combined data saved to: {output_path}")


if __name__ == "__main__":
    classes = [0,1,2,3,"",4,20]
    print(names_from_classes(classes))
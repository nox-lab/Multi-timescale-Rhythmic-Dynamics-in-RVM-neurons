import pandas as pd
import re
import sys

REQUIRED_COLUMNS = [
    "id", "weight", "weight_kg", "sex", "age_range",
    "session_type", "dataset_id", "dataset_name",
    "cells", "animal_id", "session_id", "experimenter"
]

AGE_RANGE_PATTERN = re.compile(r"^P\d+[DWMY]/P\d+[DWMY]$")
NAME_PATTERN = re.compile(
    r"^[A-Za-z\-]+(?:\s+[A-Za-z\-]+)*,\s*[A-Za-z]+(?:\s+[A-Za-z]+|\s+[A-Za-z]\.?)?$"
)

def validate_excel(filename):
    """
    Validates than an excel file contains enough information about the smrx or smr files
    it links to, so that the data can be standardised for DANDI.
    Required fields:
    [id	weight	weight_kg	sex	age_range	session_type	dataset_id	dataset_name	cells	animal_id	session_id	experimenter]

    """
    df = pd.read_excel(filename)

    errors = []

    # Column check
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
        return errors

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Missing values in non-empty rows
    for col in REQUIRED_COLUMNS:
        missing = df[df[col].isnull()]
        for idx in missing.index:
            errors.append(f"Row {idx+2}: missing value in column '{col}'")

    # id must be unique
    dup_ids = df[df["id"].duplicated()]["id"].unique()
    for d in dup_ids:
        errors.append(f"Duplicate id: {d}")

    # dataset_id must be integer
    for i, v in df["dataset_id"].items():
        if not float(v).is_integer():
            errors.append(f"Row {i+2}: dataset_id is not an integer ({v})")

    # sex must be M or F
    for i, v in df["sex"].items():
        if v not in ["M", "F"]:
            errors.append(f"Row {i+2}: invalid sex '{v}'")

    # session_id must be integer
    for i, v in df["session_id"].items():
        if pd.isna(v) or not isinstance(v,float):
            errors.append(f"Row {i+2}: session_id must be integer (got '{v}')")

    # dataset_name must be string
    for i, v in df["dataset_name"].items():
        if not isinstance(v, str):
            errors.append(f"Row {i+2}: dataset_name is not a string")

    # age_range ISO format
    for i, v in df["age_range"].items():
        if not AGE_RANGE_PATTERN.match(str(v)):
            errors.append(f"Row {i+2}: invalid age_range '{v}'")

    # experimenter name format
    for i, v in df["experimenter"].items():
        name = str(v).strip() # remove trailing whitespace
        if not NAME_PATTERN.match(name):
            errors.append(f"Row {i+2}: invalid experimenter name '{v}'")

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_excel.py yourfile.xlsx")
        sys.exit(1)

    filename = sys.argv[1]
    problems = validate_excel(filename)

    if problems:
        print("Validation failed:\n")
        for p in problems:
            print(" -", p)
    else:
        print("File is valid!")

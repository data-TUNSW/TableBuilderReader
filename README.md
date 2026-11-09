# Table Builder Reader

This module provides a flexible and robust interface for reading, cleaning, and processing ABS TableBuilder CSV outputs, with optional support for geospatial joins and advanced percentage calculations.

## Installation

Add as a git submodule in your project:
```
git submodule add https://github.com/data-TUNSW/TableBuilderReader lib/abs_methods
```

Then add the module to your Python path:
```python
import sys
sys.path.insert(0, "lib/abs_methods/src")
```

Or clone directly:
```
git clone https://github.com/data-TUNSW/TableBuilderReader
```

## Features

- **Automatic Variable Detection:**  
  Infers variables, filters, and counts from the file name and file content.
- **Flexible Cleaning:**  
  Removes totals, migratory, and overseas visitor rows as needed.
- **Geospatial Support:**  
  Optionally joins ABS shapefiles (POA or LGA) for spatial analysis.
- **Category Grouping:**  
  Supports grouping categories for custom aggregations.
- **Percentage and Standardisation:**  
  Calculates category percentages, normalised and standardised values, and weighted percentiles.
- **Processed File Saving:**  
  Optionally saves cleaned and processed data to CSV, with overwrite protection.
- **Custom Filtering and Grouping:**  
  Allows for custom filters and groupby operations on the data.

## Usage

```python
from abs_methods.table_builder_reader import TableBuilderReader

reader = TableBuilderReader(
    file_name="path/to/your_tablebuilder_file.csv",
    processed_file_name="path/to/processed_file.csv",
    save_processed_file=True,
    shapefile=True,  # Set to False if you don't want to join geographies
    percentage_categories=["Rented", "Owned"],
    category_grouping={"Tenure": ["Rented", "Owned"]},
    groupby_columns=["POA"],
    # ...other options...
)

# Access the cleaned DataFrame
df = reader.df

# Access the percentage DataFrame (if calculated)
df_percentage = reader.df_percentage
```

## Key Arguments

- `file_name`: Path to the TableBuilder CSV file.
- `processed_file_name`: Path to save the processed CSV.
- `save_processed_file`: Whether to save the processed file.
- `shapefile`: Whether to join ABS shapefiles for POA/LGA.
- `poa_shapefile_path`: Path to the ABS POA shapefile (required if `shapefile=True` and data includes POA).
- `lga_shapefile_path`: Path to the ABS LGA shapefile (required if `shapefile=True` and data includes LGA).
- `percentage_categories`: List of categories to calculate percentages for.
- `category_grouping`: Dictionary for grouping categories.
- `groupby_columns`: Columns to group by and aggregate.
- `min_max_normalisation`, `standardised`: Control normalisation and standardisation of percentages.
- `column_variable`: The variable laid across the columns (e.g. `"LFSP"`). It can sit anywhere
  in the file name; the long frame always puts it last, after the row variables.

The count column is named from the dataset code in the file name: `TBP` Persons, `TBP15`
Persons over 15, `TBD` Dwellings, `TBF` Families.

## ACTEID extracts

Tables from the Australian Census and Temporary Entrants datasets (title line "Weighted
count of Person level") are recognised from their title lines and read by a separate path:

- The geography label ("Postal Areas (EN)", "POAUCD Postal Areas (EN)", "Main Statistical
  Area Structure (Main ASGS) (EN)") is matched to the file name by position, so name the
  file with the usual code (`POA`, `STE`).
- Counts are weighted decimals. Each count's RSE is kept in an `RSE` column after the
  count; the Annotations columns are dropped.
- The column variable is read from the header, so `column_variable` is optional. If given,
  it must match the file.
- Total, Migratory and Overseas visitor rows and columns are dropped as for Census tables.
- A category label can repeat under different parent groups. The 2021 level-2 visa table
  (`VISAP Visa Subclass`) lists "Skilled Temporary Resident" under Temporary Skilled and
  again under Temporary Other. The file gives no hierarchy, only a flat list, so the groups
  come from `ACTEID_PARENT_GROUPS` in `table_builder_reader.py`, matched in file order.
  A file with repeated labels gets a `<variable> group` column just before the variable
  (`VISAP group`), holding the level-1 names used by the level-1 table ("Temporary
  Skilled"). Files with no repeated labels, including the 2016 tables, are unchanged.
  Repeats in a variable not listed there raise a warning and keep their shared label.

```python
reader = TableBuilderReader("2021_TBP_STE_SEXP_VISAP_LFSP_TENLLD-REA.csv")
reader.df.columns  # STE, SEXP, VISAP group, VISAP, LFSP, Persons, RSE
```

## Requirements

- Python 3.7+
- pandas
- numpy
- geopandas
- (optional) ABS shapefiles for POA/LGA

Install dependencies:
```bash
pip install pandas numpy geopandas
```

## Notes

- The class is designed for ABS TableBuilder outputs but can be adapted for similar tabular data.
- For geospatial joins, ensure the relevant ABS shapefiles are available at the specified paths.
- The class will raise errors if required files are missing or if processed files would be overwritten (unless `overwrite_processed_file=True`).

---

*Developed for streamlined, reproducible ABS TableBuilder
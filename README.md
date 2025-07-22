# Table Builder Reader

This module provides a flexible and robust interface for reading, cleaning, and processing ABS TableBuilder CSV outputs, with optional support for geospatial joins and advanced percentage calculations.

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
from table_builder_reader import table_builder_reader

reader = table_builder_reader(
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
- `percentage_categories`: List of categories to calculate percentages for.
- `category_grouping`: Dictionary for grouping categories.
- `groupby_columns`: Columns to group by and aggregate.
- `min_max_normalisation`, `standardised`: Control normalisation and standardisation of percentages.

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
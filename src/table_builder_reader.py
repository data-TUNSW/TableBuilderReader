import csv
import pandas as pd
import geopandas as gpd
from functools import reduce
import os


pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np


def is_list_of_lists(lst):
    """
    Check if the input is a list of lists.
    """
    return isinstance(lst, list) and all(isinstance(i, list) for i in lst)


def weighted_std(series, weights):
    """
    Calculates the weighted standard deviation of a Pandas Series.

    Args:
        series (pd.Series): The data series.
        weights (pd.Series): The weights corresponding to each data point.

    Returns:
        float: The weighted standard deviation.
    """
    df = pd.DataFrame({"series": series, "weights": weights}).dropna()
    series = df["series"]
    weights = df["weights"]

    average = np.average(series, weights=weights)
    variance = np.average((series - average) ** 2, weights=weights)
    return np.sqrt(variance)


def weighted_standardised(series, weights):
    """
    Calculates the weighted standardised value of a Pandas Series.

    Args:
        series (pd.Series): The data series.
        weights (pd.Series): The weights corresponding to each data point.

    Returns:
        pd.Series: The weighted standardised values.
    """
    df = pd.DataFrame({"series": series, "weights": weights}).dropna()
    series_dummy = df["series"]
    weights_dummy = df["weights"]

    mean = np.average(series_dummy, weights=weights_dummy)
    std_dev = weighted_std(series_dummy, weights_dummy)
    return (series - mean) / std_dev if std_dev != 0 else series - mean


class TableBuilderReader:
    """A class to read and process ABS TableBuilder files.
    This class handles the reading of TableBuilder files, extracting variables,
    applying filters, and calculating percentages. It can also join shapefiles
    for geographical data.
    Attributes:
        file_name (str): The name of the TableBuilder file.
        total (bool): Whether to include total rows.
        migratory (bool): Whether to include migratory data.
        overseas (bool): Whether to include overseas visitor data.
        infer_from_file_name (bool): Whether to infer variables from the file name.
        shapefile (bool): Whether to join shapefiles.
        geog_ff (bool): Whether to forward fill geographical data.
        clean_poa (bool): Whether to clean POA data.
        verify_file_name (dict): A dictionary to verify the file name.
        expected_footer_strings (list): A list of expected footer strings.
        percentage_categories (list): Categories for percentage calculations.
        percentage_percentile (bool): Whether to calculate weighted percentiles.
        percentage_name (str): The name for the percentage column.
        min_max_normalisation (bool): Whether to apply min-max normalisation.
        standardised (bool): Whether to standardise the percentage values.
        category_grouping (dict): A dictionary for grouping categories.
        category_group_name (str): The name for the category group.
        multivariable (bool): Whether to handle multiple variables.
        add_filter (dict): Additional filters to apply to the dataframe.
        groupby_columns (list): Columns to group by in the dataframe.
        save_processed_file (bool): Whether to save the processed dataframe to a file.
        overwrite_processed_file (bool): Whether to overwrite the processed file.
        processed_file_name (str): The name of the processed file to save.
        variables (dict): A dictionary of variables extracted from the file.
        filters (dict): A dictionary of filters applied to the dataframe.
        count (str): The count type (e.g., Dwellings, Persons).
        count_code (str): The code for the count type (e.g., TBD, TBP).
        skipfooter (int): The number of footer rows to skip when reading the file.
        footer_rows (list): A list of footer row indices.
        variable_row (int): The row index where variables are defined in the file.
        found_data (dict): A dictionary to verify if the file contains expected data.
        df (pd.DataFrame): The main dataframe containing the processed data.
        df_percentage (pd.DataFrame): A dataframe containing percentage calculations.
        shapefile_df (gpd.GeoDataFrame): A GeoDataFrame for geographical data.
    """
    def __init__(
        self,
        file_name,
        total=False,
        migratory=False,
        overseas=False,
        infer_from_file_name=True,
        shapefile=True,
        geog_ff=True,
        clean_poa=True,
        verify_file_name={
            "TBD": {"Expected string": "Counting Dwellings", "Range": range(1, 5)},
            "TBP": {"Expected string": "Counting Persons", "Range": range(1, 5)},
            "TBP15": {
                "Expected string": "Counting Persons over 15",
                "Range": range(1, 5),
            },
            "LGA": {"Expected string": "LGA (EN)", "Range": range(1, 5)},
            "POA": {"Expected string": "POA (EN)", "Range": range(1, 5)},
        },
        expected_footer_strings=[
            "Data source: Census of Population and Housing",
            "Copyright Commonwealth of Australia",
            "INFO",
            "ABS data licensed under Creative Commons",
        ],
        percentage_categories=None,  # None or a list of categories to filter by
        percentage_percentile=True,
        percentage_name=None,
        min_max_normalisation=True,
        standardised=True,
        category_grouping=None,
        category_group_name=None,
        multivariable=False,
        add_filter=None,
        groupby_columns=None,
        save_processed_file=True,
        overwrite_processed_file=False,
        processed_file_name=None,
        **kwargs,
    ):
        self.full_file_name = file_name
        self.file_name = file_name.split("/")[-1]
        self.__dict__.update(locals())
        self.__dict__.update(kwargs)
        self.variables = {}  # {"LGA": ["Albury", "Adelaide", ...], "HIED": []}
        self.filters = {}  # {"TEND": "Rented"}
        self.count = ""  # Dwellings, Persons etc
        self.count_code = ""  # TBD, TBP, TBP15 etc
        self.skipfooter = 0
        self.footer_rows = []
        self.variable_row = None
        self.found_data = self.verify_file_name
        self.df = None
        self.df_percentage = None
        self.shapefile = shapefile
        self.shapefile_df = None
        if self.infer_from_file_name:
            self.filtered_variables()
            self.set_count()
            self.set_variables()
            self.detect_variables_row()
            self.detect_footer_row()
            self.df = pd.read_csv(
                self.full_file_name,
                skiprows=self.variable_row,
                skipfooter=self.skipfooter,
                names=list(self.variables.keys()) + [self.count],
                header=0,
                index_col=False,
                engine="python",
            )
            # Cleaning
            if self.geog_ff:
                self.df.iloc[:, 0] = self.df.iloc[:, 0].ffill()
            if self.multivariable:
                self.df.iloc[:, 1] = self.df.iloc[:, 1].ffill()
            if not self.total:
                self.df = self.df[~(self.df.iloc[:, 0] == "Total")]
            if not self.migratory:
                self.df = self.df[~(self.df.iloc[:, 0].str.contains("Migratory"))]
            if not self.overseas:
                self.df = self.df[
                    ~(self.df.iloc[:, 1].astype(str).str.contains("Overseas visitor"))
                ]
            if self.clean_poa:
                if "POA" in self.variables.keys():
                    self.df["POA"] = self.df["POA"].str.extract(r"(\d{4})")
            if self.save_processed_file:
                if self.processed_file_name is None:
                    raise ValueError(
                        "processed_file_name must be provided if save_processed_file"
                        + " is True."
                    )
                if not overwrite_processed_file and os.path.exists(
                    self.processed_file_name
                ):
                    raise FileExistsError(
                        f"Processed file {self.processed_file_name} already exists. Set"
                        + " overwrite_processed_file=True to overwrite."
                    )
                self.df.to_csv(self.processed_file_name, index=False)
            if self.add_filter is not None:
                for column, filter_condition in self.add_filter.items():
                    if column in self.df.columns:
                        try:
                            self.df = pd.eval(
                                f"self.df[self.df['{column}'] {filter_condition}]"
                            )
                        except Exception as e:
                            raise ValueError(
                                f"Error applying filter {filter_condition} on column"
                                + f" {column}: {e}\nEvaluated expression: "
                                + f"self.df[self.df[{column}] {filter_condition}]"
                            )
            if self.groupby_columns is not None:
                if np.array(
                    [col in self.df.columns for col in self.groupby_columns]
                ).all():
                    self.df = (
                        self.df.groupby(self.groupby_columns)[self.count]
                        .sum()
                        .reset_index()
                    )
                    self.variables = {
                        key: value
                        for key, value in self.variables.items()
                        if key in self.df.columns
                    }
                else:
                    raise ValueError(
                        f"Group by column {self.groupby_columns} not found in dataframe"
                        + f" columns: {self.df.columns}"
                    )
            # Calculating the percentage of each category in the dataframe
            if self.category_grouping is not None:
                var = list(self.variables)[-1]
                if self.category_group_name is None:
                    category_group_name = var
                geog = list(self.variables.keys())[0]
                self.df[category_group_name] = self.df.apply(
                    lambda x: [
                        k for k, v in self.category_grouping.items() if x[var] in v
                    ][0],
                    axis=1,
                )
                self.df = (
                    self.df.groupby([geog, category_group_name])[self.count]
                    .sum()
                    .reset_index()
                )
            if self.percentage_categories is not None:
                self.df_percentage = self.percentage_in_categories(
                    categories=percentage_categories,
                    percentage_name=self.percentage_name,
                    min_max_normalisation=self.min_max_normalisation,
                    standardised=self.standardised,
                )
                if self.percentage_percentile:
                    self.df_percentage = self.weighted_percentile(
                        data=self.df_percentage,
                    )
            if self.shapefile:
                self.join_shapefile()

    def set_count(self):
        if "TBD" in self.file_name:
            self.count = "Dwellings"
            self.count_code = "TBD"
        elif "TBP" in self.file_name:
            self.count = "Persons"
            self.count_code = "TBP"
        elif "TBP15" in self.file_name:
            self.count = "Persons over 15"
            self.count_code = "TBP15"

    # TODO: Add a check for the count
    def check_count(self):
        None

    def load_shapefile(self):
        if "POA" in self.variables.keys():
            self.shapefile_df = gpd.read_file(
                "G:/Shared drives/Data/ABS/Geography/"
                + "2021 shapefiles/POA_2021_AUST_GDA2020.shp"
            ).rename(columns={"POA_CODE21": "POA"})
        elif "LGA" in self.variables.keys():
            self.shapefile_df = gpd.read_file(
                "G:/Shared drives/Data/ABS/Geography/"
                + "2021 shapefiles/LGA_2021_AUST_GDA2020.shp"
            ).rename(columns={"LGA_CODE21": "LGA"})
        else:
            raise ValueError(
                f"No known geography types found in variables: {self.variables.keys()}"
            )

    def join_shapefile(self):
        if self.shapefile_df is None:
            self.load_shapefile()
        if "POA" in self.variables.keys():
            self.df = self.shapefile_df[["POA", "geometry"]].merge(
                self.df, how="left", left_on="POA", right_on="POA"
            )
            if self.percentage_categories is not None:
                self.df_percentage = self.shapefile_df[["POA", "geometry"]].merge(
                    self.df_percentage, how="left", left_on="POA", right_on="POA"
                )
        elif "LGA" in self.variables.keys():
            self.df = self.shapefile_df[["LGA", "geometry"]].merge(
                self.df, how="left", left_on="LGA", right_on="LGA"
            )
            if self.percentage_categories is not None:
                self.df_percentage = self.shapefile_df[["LGA", "geometry"]].merge(
                    self.df_percentage, how="left", left_on="LGA", right_on="LGA"
                )
        else:
            raise ValueError(
                f"No known geography types found in variables: {self.variables.keys()}"
            )

    def set_variables(self):
        if self.count == "":
            self.set_count()
        variables = self.file_name.split(self.count_code + "_")[-1].split("_")
        if len(self.filters) > 0:
            for filter in self.filters.keys():
                for variable in variables:
                    if filter in variable:
                        variables.remove(variable)
        self.variables = {variable: [] for variable in variables}
        if self.df is not None:
            for variable in self.variables.keys():
                if variable in self.df.columns:
                    self.variables[variable] = self.df[variable].unique().tolist()
                else:
                    raise ValueError(
                        f"Variable {variable} not found in dataframe columns:"
                        + f" {self.df.columns}"
                    )

    def filtered_variables(self):
        if "-" in self.file_name:
            self.filters[self.file_name.split("-")[0].split("_")[-1]] = (
                self.file_name.split("-")[1].split(".csv")[0]
            )

    def detect_variables_row(self):
        with open(self.full_file_name) as f:
            for i, row in enumerate(csv.reader(f)):
                if len(row) > len(self.variables.keys()):
                    row = [cell for cell in row if cell != ""]
                    truth_list = [False for _ in self.variables.keys()]
                    for j in range(len(row)):
                        if list(self.variables.keys())[j] in row[j]:
                            truth_list[j] = True
                        if all(truth_list):
                            self.variable_row = i
                            return
        if self.variable_row is None:
            raise ValueError("No variable row found. Please check the file.")

    def detect_footer_row(self):
        if self.variable_row is None:
            self.detect_variables_row()
        with open(self.full_file_name) as f:
            for i, row in enumerate(csv.reader(f)):
                if i < self.variable_row:
                    continue
                if len(row) == 0:
                    self.footer_rows.append(i)
                elif len(row) > 0:
                    if all([cell == "" for cell in row]):
                        self.footer_rows.append(i)
                    else:
                        for footer_string in self.expected_footer_strings:
                            if footer_string in row[0]:
                                self.footer_rows.append(i)
        self.skipfooter = len(self.footer_rows)

    def percentage_in_categories(
        self,
        categories,
        variable=None,
        geog=None,
        percentage_name=None,
        min_max_normalisation=True,
        standardised=True,
    ):
        """
        Returns the percentage of each category in the dataframe.
        """
        if self.df is None:
            raise ValueError("No data found. Please check the file.")
        if variable is None:
            # If no variable is provided, use the last variable in the list
            variable = list(self.variables.keys())[-1]
        if geog is None:
            # If no geog is provided, use the first variable in the list
            geog = list(self.variables.keys())[0]
        if is_list_of_lists(categories):
            results = [
                self.percentage_in_categories(
                    categories=cat,
                    variable=variable,
                    geog=geog,
                    percentage_name=None,
                    min_max_normalisation=min_max_normalisation,
                )
                for cat in categories
            ]
            return reduce(lambda df1, df2: pd.merge(df1, df2), results)
        if percentage_name is None:
            if isinstance(categories, str):
                percentage_name = f"{categories} Percentage"
            elif isinstance(categories, list):
                if len(categories) == 1:
                    percentage_name = f"{categories[0]} Percentage"
                elif len(categories) > 1:
                    percentage_name = f"{', '.join(categories)} Percentage"
            else:
                raise ValueError("categories must be a string or a list of strings.")
        if isinstance(categories, str):
            categories = [categories]
        geog_totals = self.df.groupby(geog)[self.count].sum()
        geog_categories = (
            self.df[self.df[variable].isin(categories)].groupby(geog)[self.count].sum()
        )
        result = geog_categories / geog_totals
        result = result.reset_index()
        result.columns = [geog, percentage_name]
        if min_max_normalisation:
            result["Normalised " + percentage_name] = (
                result[percentage_name] - result[percentage_name].min()
            ) / (result[percentage_name].max() - result[percentage_name].min())
        if standardised:
            result["Standardised " + percentage_name] = weighted_standardised(
                result[percentage_name], self.df[self.count]
            )
        result = result.merge(geog_totals.reset_index(), on=geog, how="left")
        if self.percentage_percentile:
            result = self.weighted_percentile(
                data=result,
                variable=percentage_name,
                weights_col=self.count,
                geog=geog,
                percentile_name=percentage_name,
            )
        return result[
            [geog]
            + [col for col in result.columns if col not in [geog, self.count]]
            + [self.count]
        ]

    def weighted_percentile(
        self,
        data,
        variable=None,
        weights_col=None,
        geog=None,
        percentile_name=None,
        quantiles=np.linspace(0.001, 1, 1000),
    ):
        """
        Returns the weighted percentile of the data.
        """
        if self.df is None:
            raise ValueError("No data found. Please check the file.")
        if variable is None:
            variable = data.columns[1]
        if geog is None:
            geog = list(self.variables.keys())[0]
        if weights_col is None:
            weights_col = self.count
        if percentile_name is None:
            percentile_name = "Weighted Percentile"
        # Calculate the weighted percentile
        missing_data = data[data[weights_col] == 0]
        data = data[data[weights_col] > 0]
        percentiles = np.quantile(
            data[variable],
            quantiles,
            method="inverted_cdf",
            weights=data[weights_col],
        )
        percentile_ranges = pd.DataFrame(
            {
                "percentile": quantiles,
                "percentile_lower": np.insert(percentiles[:-1], 0, 0),
                "percentile_upper": percentiles,
            }
        )
        for idx, row in percentile_ranges.iterrows():
            data.loc[
                data[variable].between(
                    row["percentile_lower"], row["percentile_upper"]
                ),
                f"{variable} Percentile",
            ] = row["percentile"]
        data[f"{variable} Percentile"] = data[f"{variable} Percentile"].round(10)
        return pd.concat([data, missing_data], axis=0).sort_index()

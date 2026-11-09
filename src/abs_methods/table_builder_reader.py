import csv
import pandas as pd
import geopandas as gpd
from functools import reduce
import os
import numpy as np
import warnings

# pd.options.mode.chained_assignment = None  # default='warn'
pd.options.mode.copy_on_write = True  # Removes SettingWithCopyWarning:
# Consider using pd.option_context('mode.chained_assignment', None) locally if needed.


# Normal print if kwarg debug is True else print nothing


#: State codes used to name per-state chunks of a cross-tab too big for
#: TableBuilder's 40m-cell cap, e.g. ``2021_TBD_POA_SAL-NSW_DWTD-Occupied.csv``.
#: Must match ``weightings_registry.CHUNK_STATES`` (a test checks this).
STATE_CHUNK_CODES = ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT", "OT")

#: Variables whose categories are themselves state codes. ``STATE-NSW`` is a
#: real filter (``2021_TBP_TEND_STATE-NSW.csv`` is TEND for NSW only), never a
#: chunk, so a state code after one of these is left as a filter.
STATE_VARIABLES = ("STATE", "STE")

#: Parent groups for ACTEID variables whose category labels repeat under
#: different parents. TableBuilder writes the categories as a flat list with no
#: indentation or parent rows, so the group can only come from the
#: classification, given here in the order TableBuilder lists it. Keyed by the
#: variable's full header label. Group names are the level-1 labels, so they
#: match a level-1 extract such as ``2021_TBP_POA_VISAP_TENLLD-REA.csv``.
ACTEID_PARENT_GROUPS = {
    "VISAP Visa Subclass": [
        ("Temporary Bridging", ["Aggregated bridging visas", "Bridging Visa"]),
        (
            "Temporary Student",
            [
                "AusAID/Defence Sponsored",
                "ELICOS/Post Graduate",
                "HES",
                "Masters/Doctorate",
                "Other Student",
                "Schools",
                "Student",
                "VET",
            ],
        ),
        ("Temporary Skilled", ["Skilled Temporary Resident"]),
        ("Temporary Working Holiday Maker", ["Work and Holiday", "Working Holiday Maker"]),
        (
            "Temporary Other",
            [
                "Aggregated temp visas",
                "International Relations",
                "Other Sponsored Training",
                "Recognised Graduate",
                "Skilled Temporary Resident",
                "Social/Cultural",
                "Student Guardian",
                "Temp. Other",
                "Temp. Work",
                "Training and Research",
                "Transit Visa",
            ],
        ),
        ("NZ Citizen (subclass 444)", ["NZ Citizen"]),
    ],
}


def acteid_parent_groups(name, labels):
    """Parent group of each data row for one ACTEID row variable, or None.

    ``name`` is the variable's header label ("VISAP Visa Subclass"); ``labels``
    holds, for each data row in file order, the row labels up to and including
    this variable. Returns None when no label repeats within a block (the
    usual case, so the output is unchanged). When labels repeat, each block's
    labels are matched in order against ``ACTEID_PARENT_GROUPS``, so a repeated
    label takes the group of its place in the classification. Labels outside
    the classification (Total, Not applicable, Overseas visitor) are their own
    group. Repeats in a variable with no registered classification are left
    as they are, with a warning.
    """
    # One entry per category: an outer variable's label spans many data rows.
    entries, previous = [], None
    for row in labels:
        key = tuple(row)
        if key != previous:
            entries.append(key)
            previous = key
    blocks = {}
    for key in entries:
        blocks.setdefault(key[:-1], []).append(key[-1])
    if all(len(set(block)) == len(block) for block in blocks.values()):
        return None
    hierarchy = ACTEID_PARENT_GROUPS.get(name)
    if hierarchy is None:
        warnings.warn(
            f"{name!r} repeats a category label within a block and has no parent"
            + " groups in ACTEID_PARENT_GROUPS, so the repeated rows share a label"
        )
        return None
    flat = [(group, label) for group, members in hierarchy for label in members]
    known = {label for _, label in flat}
    # Match each block's labels forward through the classification.
    positions, groups_by_entry = {}, []
    for key in entries:
        parent, label = key[:-1], key[-1]
        if label not in known:
            groups_by_entry.append(label)
            continue
        start = positions.get(parent, 0)
        i = next((j for j in range(start, len(flat)) if flat[j][1] == label), None)
        if i is None:
            raise ValueError(
                f"{name!r}: {label!r} is out of the order in ACTEID_PARENT_GROUPS,"
                + " so its parent group is ambiguous"
            )
        positions[parent] = i + 1
        groups_by_entry.append(flat[i][0])
    # Expand back to one group per data row.
    out, previous, n = [], None, -1
    for row in labels:
        key = tuple(row)
        if key != previous:
            n += 1
            previous = key
        out.append(groups_by_entry[n])
    return out


def strip_chunk_suffix(token):
    """``SAL-NSW`` -> ``SAL``; any other token is returned unchanged.

    A ``-`` in a filename token otherwise means ``<variable>-<category>``
    (a filter such as ``TEND-Rented``). A state code after the ``-`` marks a
    per-state chunk of the variable instead, so it is part of the geography,
    unless the variable is itself a state (``STATE-NSW`` stays a filter).
    """
    base, sep, suffix = token.removesuffix(".csv").partition("-")
    if (
        sep
        and suffix in STATE_CHUNK_CODES
        and base.split("(")[0].strip() not in STATE_VARIABLES
    ):
        return base
    return token


def debug_print(message, debug=False):
    """
    Prints the message if debug is True, otherwise does nothing.
    """
    if debug:
        print(message)


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
        count_code (str): The code for the count type (e.g., TBD, TBP, TBP15 etc).
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
        filters_as_columns=False,
        shapefile=False,
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
        column_variable=None,
        add_filter=None,
        groupby_columns=None,
        save_processed_file=False,
        overwrite_processed_file=False,
        processed_file_name=None,
        poa_shapefile_path=None,
        lga_shapefile_path=None,
        **kwargs,
    ):
        self.filters = {}  # {"TEND": "Rented"}
        self.__dict__.update(locals())
        self.__dict__.update(kwargs)
        # After the update above, which would otherwise put the full path back.
        self.full_file_name = file_name
        # The name alone: the variables and filters are parsed from it, so a
        # directory with "-" or "_" in it (Windows paths use "\") must not leak in.
        self.file_name = os.path.basename(file_name)
        self.variables = {}  # {"LGA": ["Albury", "Adelaide", ...], "HIED": []}
        self.count = ""  # Dwellings, Persons etc
        self.count_code = ""  # TBD, TBP, TBP15 etc
        self.skipfooter = 0
        self.footer_rows = []
        self.column_variable_row = None
        self.variable_row = None
        self.found_data = self.verify_file_name
        self.df = None
        self.df_changes = None
        self.df_percentage = None
        self.shapefile = shapefile
        self.shapefile_df = None
        self.acteid = False
        if self.infer_from_file_name:
            self.filtered_variables()
            self.set_count()
            self.set_variables()
            self.acteid = self.detect_acteid()
            if not self.acteid:
                self.detect_variables_row()
                self.detect_footer_row()
            if self.acteid:
                self.df = self.read_acteid()
            elif self.column_variable is None:
                self.df = pd.read_csv(
                    self.full_file_name,
                    skiprows=self.variable_row,
                    skipfooter=self.skipfooter,
                    names=list(self.variables.keys()) + [self.count],
                    header=0,
                    index_col=False,
                    engine="python",
                )
            else:
                self.df = pd.read_csv(
                    self.full_file_name,
                    skiprows=self.column_variable_row,
                    skipfooter=self.skipfooter,
                    header=0,
                    index_col=False,
                    engine="python",
                )
                n_row_variables = len(
                    [
                        key
                        for key in self.variables
                        if key != self.column_variable and key != self.count
                    ]
                )
                index_dummy = list(self.df.columns[:n_row_variables])
                # TableBuilder writes each row label once, on the first row of
                # its block. Fill every row variable. This used to fill only
                # columns more than half blank, so an outer variable with two
                # categories under it (POA over Male/Female) was exactly half
                # blank, stayed unfilled, and dropna below lost half the rows.
                self.df[index_dummy] = self.df[index_dummy].ffill()
                self.df = (
                    self.df.set_index(index_dummy)
                    .stack(future_stack=True)
                    .reset_index()
                    .dropna()
                )
                self.df.columns = list(self.variables.keys()) + [self.count]
            self.df_changes = {"Original": self.df.copy()}
            # Cleaning
            if self.geog_ff:
                self.df.iloc[:, 0] = self.df.iloc[:, 0].ffill()
            if self.multivariable:
                self.df.iloc[:, 1] = self.df.iloc[:, 1].ffill()
            if not self.total:
                for j in range(len(self.df.columns) - 1):
                    self.df = self.df[
                        ~(self.df.iloc[:, j].astype(str).str.contains("Total"))
                    ]
            if not self.migratory:
                for j in range(len(self.df.columns) - 1):
                    self.df = self.df[
                        ~(self.df.iloc[:, j].astype(str).str.contains("Migratory"))
                    ]
            if not self.overseas:
                for j in range(len(self.df.columns) - 1):
                    # case=False because the capitalisation is not stable across
                    # extracts: the 2016 HEAP tables label this category "Overseas
                    # Visitor" while 2016 HSCP and every 2021 table use "Overseas
                    # visitor". A case-sensitive match silently kept the 2016 HEAP
                    # rows and dropped their 2021 counterparts, so any percentage
                    # derived from both years was computed over two different
                    # populations.
                    self.df = self.df[
                        ~(
                            self.df.iloc[:, j]
                            .astype(str)
                            .str.contains("Overseas visitor", case=False)
                        )
                    ]
            if self.clean_poa:
                if "POA" in self.variables.keys():
                    self.df["POA"] = self.df["POA"].str.extract(r"(\d{4})")
            self.df_changes["Clean"] = self.df.copy()
            self.set_variables()  # Update variables after cleaning
            if self.filters_as_columns:
                for filter_var, filter_value in self.filters.items():
                    self.df[filter_var] = filter_value
            if self.save_processed_file:
                if self.processed_file_name is None:
                    raise ValueError(
                        "processed_file_name must be provided if save_processed_file"
                        + " is True."
                    )
                if not self.overwrite_processed_file and os.path.exists(
                    self.processed_file_name
                ):
                    raise FileExistsError(
                        f"Processed file {self.processed_file_name} already exists.\n"
                        + "Set overwrite_processed_file=True to overwrite."
                    )
                self.df.to_csv(self.processed_file_name, index=False)
            if self.add_filter is not None:
                self.apply_filters(add_filters=self.add_filter)
                self.df_changes["Filtered"] = self.df.copy()
            if self.groupby_columns is not None:
                self.apply_groupby_columns(groupby_columns=self.groupby_columns)
                self.df_changes["Grouped"] = self.df.copy()
            # Calculating the percentage of each category in the dataframe
            if self.category_grouping is not None:
                self.apply_category_grouping()
                self.df_changes["Category Grouped"] = self.df.copy()
            if self.percentage_categories is not None:
                self.df_percentage = self.percentage_in_categories(
                    categories=self.percentage_categories,
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

    def apply_filters(self, add_filters=None):
        """
        Applies additional filters to the dataframe.
        """
        if add_filters is None:
            add_filters = self.add_filter
        elif add_filters is not None:
            for column, filter_condition in add_filters.items():
                if column in self.df.columns:
                    try:
                        self.df = pd.eval(
                            f"self.df[self.df['{column}']{filter_condition}]"
                        )
                    except Exception:
                        try:  # Convert column to numeric if filter fails
                            self.df[column] = pd.to_numeric(
                                self.df[column], errors="coerce"
                            )
                            self.df = pd.eval(
                                f"self.df[self.df['{column}']{filter_condition}]"
                            )
                        except Exception as e:
                            raise ValueError(
                                "Error applying filter "
                                + f"{filter_condition} on column {column}: {e}\n"
                                + "Evaluated expression: "
                                + f"self.df[self.df[{column}]{filter_condition}]"
                            )
        else:
            raise ValueError("No filters provided to apply_filters method.")

    def apply_groupby_columns(self, groupby_columns=None):
        """
        Applies groupby operation on the dataframe based on specified columns.
        """
        if groupby_columns is None:
            groupby_columns = self.groupby_columns
        elif groupby_columns is not None:
            if np.array([col in self.df.columns for col in self.groupby_columns]).all():
                self.df = (
                    self.df.groupby(self.groupby_columns)[self.count]
                    .sum()
                    .reset_index()
                )
                # Update variables to match new structure
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

    def apply_category_grouping(self):
        """
        Applies category grouping to the dataframe based on the
        category_grouping dictionary.
        """
        if self.category_grouping is not None:
            var = list(self.variables)[-1]
            if self.category_group_name is None:
                category_group_name = var
            else:
                category_group_name = self.category_group_name
            geog = list(self.variables.keys())[0]
            self.df[category_group_name] = self.df.apply(
                lambda x: [k for k, v in self.category_grouping.items() if x[var] in v][
                    0
                ],
                axis=1,
            )
            self.df = (
                self.df.groupby([geog, category_group_name])[self.count]
                .sum()
                .reset_index()
            )
            # Optionally update self.variables for the new group

    #: Dataset code in the file name -> count name. Matched as a whole
    #: ``_``-separated token, so ``TBP15`` is not read as ``TBP`` and a filter
    #: or category that happens to contain a code is not the dataset.
    COUNT_CODES = {
        "TBD": "Dwellings",
        "TBP": "Persons",
        "TBP15": "Persons over 15",
        "TBF": "Families",
    }

    def set_count(self):
        tokens = self.file_name.removesuffix(".csv").split("_")
        for token in tokens:
            if token in self.COUNT_CODES:
                self.count = self.COUNT_CODES[token]
                self.count_code = token
                return

    # TODO: Add a check for the count
    def check_count(self):
        None

    def load_shapefile(self):
        if "POA" in self.variables.keys():
            self.shapefile_df = gpd.read_file(self.poa_shapefile_path).rename(
                columns={"POA_CODE21": "POA"}
            )
        elif "LGA" in self.variables.keys():
            self.shapefile_df = gpd.read_file(self.lga_shapefile_path).rename(
                columns={"LGA_CODE21": "LGA"}
            )
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
        """Sets the variables attribute based on the file name and filters."""
        if self.count == "":
            self.set_count()
        # Split after count_code, then split by "_"
        variables = (
            self.file_name.replace(".csv", "")
            .split(self.count_code + "_")[-1]
            .split("_")
        )
        variable_names = []
        variable_descriptions = []
        for var in variables:
            var = strip_chunk_suffix(var)
            if "(" in var and ")" in var:
                # Extract base name and bracket content
                base = var.split("(")[0]
                start_bracket = var.find("(") + 1
                end_bracket = var.find(")")
                bracket = var[start_bracket:end_bracket]
                variable_names.append(base)
                variable_descriptions.append(f"{base} {bracket}")
            else:
                variable_names.append(var)
                variable_descriptions.append(var)
        # Remove filtered variables from the list of variables
        if len(self.filters) > 0:
            for filter in self.filters.keys():
                for i, variable in enumerate(variable_names):
                    if filter in variable:
                        variable_names.pop(i)
                        variable_descriptions.pop(i)
                        break
        # Restructure: each variable is a dict with "data" and "description"
        self.variables = {
            variable_names[i]: {"data": [], "description": variable_descriptions[i]}
            for i in range(len(variable_names))
        }
        # The column variable is stacked into the last position of the long
        # frame whatever its place in the file name, so it goes last here too
        # and the frame's columns are labelled in stacked order.
        if self.column_variable in self.variables:
            self.variables[self.column_variable] = self.variables.pop(
                self.column_variable
            )
        if self.df is not None:
            for variable in self.variables.keys():
                if variable in self.df.columns:
                    self.variables[variable]["data"] = (
                        self.df[variable].unique().tolist()
                    )
                else:
                    raise ValueError(
                        f"Variable {variable} not found in dataframe columns:"
                        + f" {self.df.columns}"
                    )

    def filtered_variables(self):
        if len(self.filters) == 0:
            if "-" in self.file_name:
                self.filters = {
                    var[0]: var[1].replace(".csv", "")
                    for var in [
                        key.split("-")
                        for key in self.file_name.split("_")
                        if "-" in key and strip_chunk_suffix(key) == key
                    ]
                }

    #: Title-line text that marks an Australian Census and Temporary Entrants
    #: (ACTEID) extract rather than a Census one.
    ACTEID_MARKERS = (
        "Australian Census and Temporary Entrants",
        "Weighted count of Person level",
    )

    def detect_acteid(self, limit_rows=12):
        """True if the title lines mark the file as an ACTEID extract."""
        with open(self.full_file_name, newline="", encoding="utf-8-sig") as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= limit_rows:
                    break
                if row and any(m in row[0] for m in self.ACTEID_MARKERS):
                    return True
        return False

    def read_acteid(self):
        """Long frame from an ACTEID extract: row variables, the column
        variable (if any), the count and its RSE.

        ACTEID extracts label the geography differently from Census ones
        ("Postal Areas (EN)", "POAUCD Postal Areas (EN)", "Main Statistical Area
        Structure (Main ASGS) (EN)"), so the row variables are matched to the
        file name by position, not by label. Counts are weighted decimals, each
        followed by "- RSE" and "- Annotations" columns: the RSE is kept as a
        column and the Annotations are dropped. The data rows end at the first
        blank line, so the Symbol / Description footer block is never read.

        The column variable is read from the header cell above the row labels
        ("Counting" means there is none). If ``column_variable`` was given it
        must match; if not, it is set from the header.
        """
        with open(self.full_file_name, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        # The row-variable name line is the first with a non-empty first cell
        # that follows a line starting with an empty cell (the column header).
        for i in range(1, len(rows)):
            prev, cur = rows[i - 1], rows[i]
            if prev and prev[0] == "" and len(prev) > 2 and cur and cur[0] != "":
                header, start = prev, i + 1
                n_rows = len([c for c in cur if c != ""])
                break
        else:
            raise ValueError(f"No header row found in ACTEID file {self.file_name}")
        label = header[n_rows - 1]
        if label == "Counting":
            column_variable = None
        else:
            column_variable = label.split(" ")[0]
            if column_variable not in self.variables:
                raise ValueError(
                    f"Column variable {column_variable!r} in {self.file_name} is not "
                    + f"in the file name variables {list(self.variables)}"
                )
        if self.column_variable is not None and self.column_variable != column_variable:
            raise ValueError(
                f"column_variable={self.column_variable!r} but {self.file_name} has "
                + f"{column_variable!r} in its columns"
            )
        if column_variable is not None and self.column_variable is None:
            self.column_variable = column_variable
            self.set_variables()  # puts the column variable last
        row_variables = [v for v in self.variables if v != column_variable]
        if len(row_variables) != n_rows:
            raise ValueError(
                f"{self.file_name} has {n_rows} row variables in the file but "
                + f"{row_variables} in its name"
            )
        # Count columns, each with the position of its RSE column (or None).
        count_columns = []
        for j in range(n_rows, len(header)):
            name = header[j]
            if name == "" or name.endswith(" - RSE") or name.endswith(" - Annotations"):
                continue
            rse = j + 1 if header[j + 1 : j + 2] == [f"{name} - RSE"] else None
            count_columns.append((j, name, rse))

        def number(row, j):
            if j is None or j >= len(row) or row[j] == "":
                return np.nan
            return float(row[j])

        records, last, row_labels = [], [""] * n_rows, []
        for row in rows[start:]:
            if not row or all(cell == "" for cell in row):
                break
            last = [row[j] if row[j] != "" else last[j] for j in range(n_rows)]
            row_labels.append(last)
            for j, name, rse in count_columns:
                labels = last + ([name] if column_variable else [])
                records.append((*labels, number(row, j), number(row, rse)))
        df = pd.DataFrame(
            records, columns=list(self.variables) + [self.count, "RSE"]
        )
        row_names = [c for c in cur if c != ""]
        for k, variable in enumerate(row_variables):
            groups = acteid_parent_groups(
                row_names[k], [labels[: k + 1] for labels in row_labels]
            )
            if groups is not None:
                df.insert(
                    df.columns.get_loc(variable),
                    f"{variable} group",
                    np.repeat(groups, len(count_columns)),
                )
        return df

    def detect_variables_row(self, limit_rows=100):
        truth_list = {
            var: False for var in self.variables.keys() if var != self.column_variable
        }
        with open(self.full_file_name) as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= limit_rows:
                    break
                # Check if the row is long enough to contain all variables
                if (len(row) > len(truth_list.values())) and i < limit_rows:
                    row = [cell for cell in row if cell != ""]
                    # Loop through all cells in row
                    for j in range(len(row)):
                        if (
                            self.column_variable_row is None
                            and self.column_variable is not None
                            and self.column_variable in row[j]
                        ):
                            self.column_variable_row = i
                        # Check if the variables are in the row
                        for var in truth_list.keys():
                            if var in row[j]:
                                truth_list[var] = True
                        # If all variables are found in the row, set the variable_row
                        if all(truth_list.values()):
                            self.variable_row = i
                            return
        if self.variable_row is None:
            raise ValueError(
                "No variable row found. Please check the file.\n"
                + f"Truth list {truth_list}\n"
                + f"Variables: {self.variables.keys()}\n"
                + f"Column variable: {self.column_variable}\n"
                + f"Column variable row: {self.column_variable_row}\n"
                + f"File name: {self.file_name}"
            )

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
        debug=False,
    ):
        """
        Returns the percentage of each category in the dataframe.
        """
        if self.df is None:
            raise ValueError("No data found. Please check the file.")
        if variable is None:
            # If no variable is provided, use the last variable in the list
            variable = list(self.variables.keys())[-1]
            debug_print(f"No variable found, Using variable: {variable}", debug=debug)
        if geog is None:
            # If no geog is provided, use the first variable in the list
            geog = list(self.variables.keys())[0]
            debug_print(f"No geography found, Using geography: {geog}", debug=debug)
        # When accessing unique values, use self.variables[variable]["data"]
        # When accessing description, use self.variables[variable]["description"]
        if is_list_of_lists(categories):
            # If categories is a list of lists, calculate percentage for each sublist
            debug_print(f"Categories are a list of lists: {categories}", debug=debug)
            results = [
                self.percentage_in_categories(
                    categories=cat,
                    variable=variable,
                    geog=geog,
                    percentage_name=None,
                    min_max_normalisation=min_max_normalisation,
                    standardised=standardised,
                )
                for cat in categories
            ]
            return reduce(lambda df1, df2: pd.merge(df1, df2), results)
        if percentage_name is None:
            # No percentage name is provided, default name based on categories
            debug_print(
                (
                    "No percentage name provided,"
                    f"using default based on categories: {categories}"
                ),
                debug=debug,
            )
            if isinstance(categories, str):
                percentage_name = f"{categories} Percentage"
                debug_print(f"Using percentage name: {percentage_name}", debug=debug)
            elif isinstance(categories, list):
                debug_print(f"Using categories: {categories}", debug=debug)
                if len(categories) == 1:
                    percentage_name = f"{categories[0]} Percentage"
                    debug_print(
                        f"Using single category percentage name: {percentage_name}",
                        debug=debug,
                    )
                elif len(categories) > 1:
                    percentage_name = f"{', '.join(categories)} Percentage"
                    debug_print(
                        f"Using multiple categories percentage name: {percentage_name}",
                        debug=debug,
                    )
            else:
                raise ValueError("categories must be a string or a list of strings.")
        if isinstance(categories, str):
            categories = [categories]
            debug_print(
                f"Categories is a string, converted to list: {categories}", debug=debug
            )
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
            debug_print(
                f"Min-max normalisation applied to {percentage_name}", debug=debug
            )
        if standardised:
            # Merge weights from the original dataframe based on the geography column
            weights = (
                self.df.groupby(geog)[self.count]
                .sum()
                .reset_index()
                .rename(columns={self.count: self.count + "_weights"})
            )
            debug_print(weights, debug=debug)
            debug_print(result, debug=debug)
            result = result.merge(weights, on=geog, suffixes=("", "_weights"))
            debug_print(result, debug=debug)
            result["Standardised " + percentage_name] = weighted_standardised(
                result[percentage_name], result[self.count + "_weights"]
            )
            result = result.drop(columns=[self.count + "_weights"])
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

    def __repr__(self):
        return str(self.__dict__)

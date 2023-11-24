import pandas as pd
import numpy as np
import sqlite3
import re
import warnings

class piplyr:
    """
    A class providing dplyr-like data manipulation capabilities for pandas DataFrames.
    """

    def __init__(self, df):
        """
        Initializes the piplyr class with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): A pandas DataFrame to be manipulated.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df
        self.grouped = None

    def group_by(self, *group_vars):
        """
        Groups the DataFrame by specified columns.

        Args:
            group_vars: Columns to group by. Multiple columns can be specified.

        Returns:
            self: The piplyr object with the DataFrame grouped.
        """
        if not all(col in self.df.columns for col in group_vars):
            raise ValueError("One or more grouping columns not found in DataFrame")
        self.grouped = self.df.groupby(list(group_vars))
        return self

    def sort_by(self, column, ascending=True):
        """
        Sorts the DataFrame by a specified column.

        Args:
            column: The column to sort by.
            ascending: Whether to sort in ascending order (default is True).

        Returns:
            self: The piplyr object with the DataFrame sorted.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        self.df = self.df.sort_values(by=column, ascending=ascending)
        return self

    def select(self, *columns):
        """
        Selects specified columns from the DataFrame.

        Args:
            columns: A list of column names to keep in the DataFrame.

        Returns:
            self: The modified piplyr object with only the selected columns.
        """
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        self.df = self.df[list(columns)]
        return self

    def drop_col(self, *columns):
        """
        Drops specified columns from the DataFrame.

        Args:
            columns: A list of column names to drop from the DataFrame.

        Returns:
            self: The modified piplyr object with specified columns removed.
        """
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        self.df = self.df.drop(columns=list(columns))
        return self

    def rename_col(self, rename_dict):
        """
        Renames columns in the DataFrame.

        Args:
            rename_dict: A dictionary mapping old column names to new column names.

        Returns:
            self: The modified piplyr object with columns renamed.
        """
        self.df = self.df.rename(columns=rename_dict)
        return self

    def filter_row(self, condition):
        """
        Filters rows based on a given condition.

        Args:
            condition: A string of condition to filter the DataFrame rows. 
                       The condition should be in a format that can be used inside DataFrame.query().

        Returns:
            self: The modified piplyr object with rows filtered based on the condition.
        """
        self.df = self.df.query(condition)
        return self

    def mutate(self, **kwargs):
        """
        Adds new columns or modifies existing ones based on given expressions.
        When used after 'group_by', applies the mutation within each group.

        Args:
            kwargs: Key-value pairs where keys are new or existing column names 
                    and values are expressions or functions to compute their values.

        Returns:
            self: The modified piplyr object with new or modified columns.

        Examples:
            Without grouping:
            pi.mutate(new_col='A * B')

            With grouping:
            pi.group_by('C').mutate(mean_val=lambda x: x['A'].mean())
        """
        if self.grouped:
            for new_col, func in kwargs.items():
                if callable(func):
                    # Apply the function to each group and assign the result
                    self.df[new_col] = self.grouped.apply(func).reset_index(level=0, drop=True)
                else:
                    raise ValueError("With grouping, provide a callable function for mutation.")
        else:
            for new_col, expression in kwargs.items():
                self.df[new_col] = self.df.eval(expression)
        return self



    def summarize(self, **kwargs):
        """
        Performs summary/aggregation operations on the DataFrame.
        If used after 'group_by', provides aggregated statistics for each group.
        Without 'group_by', provides aggregated statistics for the entire DataFrame.

        Args:
            kwargs: Key-value pairs where keys are new column names for the aggregated values and
                    values are aggregation functions.

        Returns:
            self: The modified piplyr object with a DataFrame containing summary statistics.

        Examples:
            Without grouping:
            pi.summarize(mean_A=('A', 'mean'), sum_B=('B', 'sum'))

            With grouping:
            pi.group_by('C').summarize(mean_A=('A', 'mean'))
        """
        if self.grouped:
            self.df = self.grouped.agg(kwargs).reset_index()
        else:
            self.df = self.df.agg(kwargs)
        return self

    # ... [Other methods and existing class functionality] ...
    
    
    def sql_plyr(self, expression):
        """
        Executes an SQL query on the DataFrame.

        Args:
            expression: The SQL query to execute.

        Returns:
            self: The piplyr object with the DataFrame modified by the SQL query.
        """
        with sqlite3.connect(':memory:') as con:
            self.df.to_sql('df', con, index=False)
            self.df = pd.read_sql_query(expression, con)
        return self

    # ... [Other methods like case_when, join, count_na, etc.] ...

    def pipe(self, func, *args, **kwargs):
        """
        Allows the use of external functions in a chain.

        Args:
            func: A function to apply to the DataFrame.
            *args, **kwargs: Additional arguments and keyword arguments for the function.

        Returns:
            self: The modified piplyr object.
        """
        self.df = func(self.df, *args, **kwargs)
        return self



    def case_when(self, cases, target_var):
        """
        Applies conditions and assigns values based on them, similar to SQL's CASE WHEN.

        Args:
            cases: A list of tuples containing conditions and corresponding values.
            target_var: The name of the new or existing column to store the result.

        Returns:
            self: The modified piplyr object.
        """
        self.df[target_var] = np.nan
        for condition, value in cases:
            self.df.loc[self.df.eval(condition), target_var] = value
        return self

    def join(self, other_df, by, join_type='inner'):
        """
        Joins the current DataFrame with another DataFrame.

        Args:
            other_df: The DataFrame to join with.
            by: The column name(s) to join on.
            join_type: Type of join to perform ('inner', 'left', 'right', 'outer').

        Returns:
            self: The modified piplyr object.
        """
        if join_type not in ['inner', 'left', 'right', 'outer']:
            raise ValueError("join_type must be one of 'inner', 'left', 'right', 'outer'")
        self.df = self.df.merge(other_df, on=by, how=join_type)
        return self

    def count_na(self):
        """
        Counts the number of NA values in each column of the DataFrame.

        Returns:
            pd.Series: A Series with the count of NA values for each column.
        """
        return self.df.isna().sum()

    # ... [Other existing methods] ...

    def distinct(self, columns=None):
        """
        Removes duplicate rows in the DataFrame.

        Args:
            columns: The columns to consider for identifying duplicates. 
                     If None, all columns are considered.

        Returns:
            self: The modified piplyr object.
        """
        self.df = self.df.drop_duplicates(subset=columns)
        return self

    def skim(self):
        """
        Provides a summary of the DataFrame's statistics.

        Returns:
            pd.DataFrame: A DataFrame containing summary statistics for each column.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = {
                'Types': self.df.dtypes,
                'Missing Values': self.df.isna().sum(),
                'Unique Values': self.df.nunique(),
                'Min': self.df.min(),
                'Max': self.df.max(),
                'Mean': self.df.mean(),
                'Std': self.df.std(),
                '25%': self.df.quantile(0.25),
                '50%': self.df.quantile(0.50),
                '75%': self.df.quantile(0.75)
            }
            return pd.DataFrame(stats)

    def pivot_longer(self, cols, id_vars=None, var_name='variable', value_name='value'):
        """
        Transforms the DataFrame from a wide format to a long format.

        Args:
            cols: Columns to unpivot.
            id_vars: Columns to leave unchanged (identifier variables).
            var_name: Name of the new column that will contain the variable names.
            value_name: Name of the new column that will contain the values.

        Returns:
            self: The modified piplyr object.
        """
        self.df = pd.melt(self.df, id_vars=id_vars, value_vars=cols, var_name=var_name, value_name=value_name)
        return self

    def pivot_wider(self, index, columns, values):
        """
        Transforms the DataFrame from a long format to a wide format.

        Args:
            index: Column(s) to use as index (identifier variables).
            columns: Column whose unique values will become new columns in the wide format.
            values: Column(s) that will be spread across the new columns.

        Returns:
            self: The modified piplyr object.
        """
        self.df = self.df.pivot(index=index, columns=columns, values=values)
        return self

    def clean_names(self):
        """
        Cleans and standardizes column names by converting them to lowercase and replacing 
        non-alphanumeric characters with underscores.

        Returns:
            self: The modified piplyr object.
        """
        self.df.columns = [re.sub('[^0-9a-zA-Z]+', '_', col).lower() for col in self.df.columns]
        return self
    

    def separate(self, col, into, sep=None, remove=False, extra='warn'):
        """
        Separates a column into multiple columns based on a separator.

        Args:
            col: Column name to be separated.
            into: List of new column names after separation.
            sep: Separator to split the column (default: split on whitespace).
            remove: Flag to remove the original column (default: False).
            extra: Specifies behavior if there are extra splits. Options are 'drop', 'merge', and 'warn'.

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        split_cols = self.df[col].str.split(sep, expand=True)
        num_cols = len(into)

        if split_cols.shape[1] > num_cols:
            if extra == 'drop':
                split_cols = split_cols.iloc[:, :num_cols]
            elif extra == 'merge':
                split_cols.iloc[:, num_cols-1] = split_cols.iloc[:, num_cols-1:].apply(lambda x: sep.join(x.dropna().astype(str)), axis=1)
                split_cols = split_cols.iloc[:, :num_cols]
            elif extra == 'warn':
                warnings.warn("Number of splits exceeds the length of 'into'; extra splits are being dropped.")

        self.df[into] = split_cols
        if remove:
            self.df = self.df.drop(columns=[col])
        return self

    def str_pad(self, column, width, side='left', pad=" "):
        """
        Pads strings in a DataFrame column to a specified width.

        Args:
            column: The column to pad.
            width: The width to pad the strings to.
            side: The side to pad on ('left' or 'right').
            pad: The character used for padding (default: space).

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        if side not in ['left', 'right']:
            raise ValueError("Side must be either 'left' or 'right'")

        if side == 'left':
            self.df[column] = self.df[column].astype(str).str.pad(width, side='left', fillchar=pad)
        else:  # side == 'right'
            self.df[column] = self.df[column].astype(str).str.pad(width, side='right', fillchar=pad)
        return self

    def str_sub(self, pattern, replacement):
        """
        Replaces a pattern in strings with a replacement string.

        Args:
            pattern: The regex pattern to replace.
            replacement: The replacement string.

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        self.df = self.df.applymap(lambda x: re.sub(pattern, replacement, str(x)) if isinstance(x, str) else x)
        return self

    def str_extract(self, pattern, col=None):
        """
        Extracts a pattern from a string column.

        Args:
            pattern: The regex pattern to extract.
            col: The column to apply extraction. If None, applies to all string columns.

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        if col:
            self.df[col + '_extracted'] = self.df[col].str.extract(pattern)
        else:
            for c in self.df.columns:
                if self.df[c].dtype == object:
                    self.df[c + '_extracted'] = self.df[c].str.extract(pattern)
        return self

    def str_detect(self, col, pattern):
        """
        Detects if a pattern exists in a string column.

        Args:
            col: The column to check for the pattern.
            pattern: The regex pattern to detect.

        Returns:
            self: The piplyr object with a new column indicating detection.
        """
        self.df[col + '_detected'] = self.df[col].str.contains(pattern, na=False)
        return self

    def fct_lump(self, column, n=10, other_level='Other'):
        """
        Lumps less frequent levels of a categorical column into an 'Other' category.

        Args:
            column: The name of the categorical column.
            n: The minimum count to not be lumped into 'Other'.
            other_level: The name for the lumped category (default: 'Other').

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        value_counts = self.df[column].value_counts()
        self.df[column] = np.where(self.df[column].isin(value_counts.index[value_counts >= n]), self.df[column], other_level)
        return self

    def fct_infreq(self, column, frac=0.01, other_level='Other'):
        """
        Lumps infrequent levels of a categorical column based on a fraction of total occurrences.

        Args:
            column: The name of the categorical column.
            frac: Fraction of total occurrences to be considered infrequent.
            other_level: The name for the lumped category (default: 'Other').

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        value_counts = self.df[column].value_counts(normalize=True)
        self.df[column] = np.where(self.df[column].isin(value_counts.index[value_counts >= frac]), self.df[column], other_level)
        return self

    def fct_relevel(self, column, ref_level, after=True):
        """
        Reorders levels of a categorical column, moving a specified level to the first or last.

        Args:
            column: The name of the categorical column.
            ref_level: The reference level to move.
            after: Whether to move the reference level after the other levels.

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        self.df[column] = pd.Categorical(self.df[column], categories=self.df[column].unique(), ordered=True)
        if ref_level not in self.df[column].cat.categories:
            raise ValueError(f"Reference level '{ref_level}' not found in column '{column}'")
        if after:
            new_order = [cat for cat in self.df[column].cat.categories if cat != ref_level] + [ref_level]
        else:
            new_order = [ref_level] + [cat for cat in self.df[column].cat.categories if cat != ref_level]
        self.df[column] = self.df[column].cat.reorder_categories(new_order)
        return self

    def fct_recode(self, column, recode_dict, drop_unused=False):
        """
        Recodes levels of a categorical column.

        Args:
            column: The name of the categorical column.
            recode_dict: Dictionary mapping old levels to new levels.
            drop_unused: Whether to drop unused categories after recoding.

        Returns:
            self: The piplyr object with the DataFrame modified.
        """
        if not all(level in self.df[column].cat.categories for level in recode_dict.keys()):
            raise ValueError("One or more levels to recode not found in column categories")
        self.df[column] = self.df[column].cat.rename_categories(recode_dict)
        if drop_unused:
            self.df[column] = self.df[column].cat.remove_unused_categories()
        return self

    # ... [Other methods and existing class functionality] ...
    

    def __call__(self, df):
        """
        Allows the piplyr object to be called with a new DataFrame, replacing the current one.

        Args:
            df: A new pandas DataFrame to replace the current one.

        Returns:
            self: The piplyr object with the new DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df
        self.grouped = None
        return self

    def __repr__(self):
        """
        Returns a string representation of the DataFrame.

        Returns:
            str: A string representation of the DataFrame.
        """
        return self.df.__repr__()

    @property
    def to_df(self):
        """
        Converts the piplyr object's DataFrame to a standard pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame contained within the piplyr object.
        """
        return pd.DataFrame(self.df)

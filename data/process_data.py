import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load_data
    load messages and categories csv file and merge to a datafram.

    :param messages_filepath: The messages.csv file
    :param categories_filepath: The categories.csv file

    :returns: A dataframe
    """
    # Load datasets.
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets.
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """clean_data
    Create a One-Hot encoding dataframe with no duplicates.

    :param df: The pure dataframe

    :returns: A One-Hot encoding dataframe
    """
    # Split categories into separate category columns.
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns.
    df.drop(labels='categories', axis=1, inplace=True)
    df = pd.concat((df, categories), axis=1)

    # Remove duplicates.
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """save_data
    Save the df dataframe to a database_filename file.

    :param df: The dataframe
    :param database_filename: A SQLite database_filename file
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

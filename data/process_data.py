import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the message text and categorization and returns them
    in a joined pandas dataframe.
    
    messages_filepath: str
        the path to the messages file to be loaded
    categories_filepath: str
        the path to the categories file to be loaded
        
    returns: pandas.DataFrame
        a pandas dataframe containing the joined categories and messages
    """
    # load the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge and return
    return pd.merge(messages, categories, how='left')

def clean_data(df):
    """
    This function applies the cleaning shown to be necessary in the sample data.
    Steps:
        - Process the 'categories' column into a usable state
            - Split into different columns on semicolon
            - Extract proper column names using regex
            - Extract the individual values using regex
            - Replace the original column with processed output
        - Deduplicate the dataframe
            - Use all columns besides ID to decide if a row is unique
            - Determine first index of each unique row
            - Filter to only the first indices (keeps 1 of each row with dupes)
        
    df: pandas.DataFrame
        the dataframe to be cleaned
    
    returns: df (pandas.DataFrame)
        the cleaned dataframe
        
    """
    ### Process the categories column
    
    # split on semicolon into different columns
    categories = df['categories'].str.split(';',expand=True)
    # extract column names from first row by removing the trailing '-1' or '-0'
    category_colnames = categories.iloc[0,:].apply(lambda x: re.sub(r"-\d","",x))
    categories.columns = category_colnames
    # extract values from each cell by removing the preceding label and -, and convert to int
    categories = categories.applymap(lambda x: int(re.sub('\w+-','',x)))
    # drop original categories column
    df.drop(columns=['categories'], inplace=True)
    # add the processed categories
    df = pd.concat([df, categories], axis=1)
    
    ### Deduplicate the dataframe
    
    # Concatenate all columns used to determine uniqueness into a new column using str.join
    df['unique_mash'] = df.loc[:, df.columns[1:]].apply(lambda x: '##'.join(x.astype(str)), axis=1)
    
    # determine indices of rows to keep
    indices_to_keep = (df
                        # pull original index into the dataframe so we can use it later
                        .reset_index()
                        # group by the unique mash
                        .groupby('unique_mash')
                        # take the first version of each unique element
                        # and extract the original pandas index
                        .first()['index']
                        # turn into a numpy array of the index values
                        .values
                       )
    
    # filter to only the desired indices and drop the unique mash temp column
    df = df[df.index.isin(indices_to_keep)].drop(columns=['unique_mash'])

    return df

def save_data(df, database_filename):
    """
    This function exports the dataframe to the specified database filename,
    putting the data in a table called 'messages'.  Pandas' df.to_sql()
    method is used, replacing data if it already exists.
    
    df: pandas.DataFrame
        the dataframe to be exported to the database
    database_filename: str
        the path to the database file
    
    returns: None
    """
    engine = create_engine('sqlite:///' + database_filename)  
    df.to_sql('messages', engine, index=False, if_exists='replace')

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
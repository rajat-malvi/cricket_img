"""
Data processing module for the Cricket Image Chatbot
"""

import os
import pandas as pd
from typing import List
from langchain.docstore.document import Document

import config

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    os.makedirs(config.CACHE_DIR, exist_ok=True)

def load_csv_data() -> pd.DataFrame:
    """
    Load the cricket image data from CSV file

    Returns:
        pd.DataFrame: DataFrame containing the cricket image data
    """
    if not os.path.exists(config.CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {config.CSV_FILE}")

    return pd.read_csv(config.CSV_FILE)

def load_reference_tables():
    """
    Load reference tables (Action, Event, Mood, Players, Sublocation)

    Returns:
        dict: Dictionary of DataFrames for each reference table
    """
    tables = {}

    # Define the reference tables
    reference_files = [
        'Action.csv',
        'Event.csv',
        'Mood.csv',
        'Players.csv',
        'Sublocation.csv'
    ]

    # Load each reference table
    for file in reference_files:
        file_path = os.path.join(config.DATA_DIR, file)
        if os.path.exists(file_path):
            table_name = file.split('.')[0].lower()
            tables[table_name] = pd.read_csv(file_path)

    return tables

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the cricket image data

    Args:
        df (pd.DataFrame): Raw DataFrame from CSV

    Returns:
        pd.DataFrame: Preprocessed DataFrame with comprehensive description
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Fill NaN values
    processed_df = processed_df.fillna('')

    # Load reference tables
    reference_tables = load_reference_tables()

    # Map IDs to names for better search context
    if 'action' in reference_tables:
        action_map = dict(zip(reference_tables['action']['action_id'],
                             reference_tables['action']['action_name']))
        if 'action_id' in processed_df.columns:
            processed_df['action_name'] = processed_df['action_id'].map(action_map)

    if 'event' in reference_tables:
        event_map = dict(zip(reference_tables['event']['event_id'],
                            reference_tables['event']['event_name']))
        if 'event_id' in processed_df.columns:
            processed_df['event_name'] = processed_df['event_id'].map(event_map)

    if 'mood' in reference_tables:
        mood_map = dict(zip(reference_tables['mood']['mood_id'],
                           reference_tables['mood']['mood_name']))
        if 'mood_id' in processed_df.columns:
            processed_df['mood_name'] = processed_df['mood_id'].map(mood_map)

    if 'sublocation' in reference_tables:
        sublocation_map = dict(zip(reference_tables['sublocation']['sublocation_id'],
                                  reference_tables['sublocation']['sublocation_name']))
        if 'sublocation_id' in processed_df.columns:
            processed_df['sublocation_name'] = processed_df['sublocation_id'].map(sublocation_map)

    if 'players' in reference_tables:
        player_map = dict(zip(reference_tables['players']['player_id'],
                             reference_tables['players']['Player Name']))
        if 'player_id' in processed_df.columns:
            # Handle multiple player IDs in a single cell (comma-separated)
            def map_player_names(player_ids):
                if not player_ids or pd.isna(player_ids):
                    return ''
                # Split by comma and handle multiple IDs
                if ',' in player_ids:
                    ids = [pid.strip() for pid in player_ids.split(',')]
                    names = [player_map.get(pid, '') for pid in ids]
                    return ', '.join([name for name in names if name])
                # Handle single ID
                return player_map.get(player_ids, '')

            processed_df['player_name'] = processed_df['player_id'].apply(map_player_names)

    # Create a comprehensive description field for embedding
    processed_df['description'] = ''

    # Start with the caption as the base description
    if 'caption' in processed_df.columns:
        processed_df['description'] = processed_df['caption']

    # Build a comprehensive description using all available columns
    for column in processed_df.columns:
        # Skip columns that are already included or not useful for description
        if column in ['description', 'combined_text', 'URL', 'File Name']:
            continue

        # Add column data to description with proper formatting
        if not pd.isna(processed_df[column]).all() and processed_df[column].astype(str).str.strip().ne('').any():
            # Format the column name for readability
            col_name = column.replace('_', ' ').title()

            # Add the column data to the description
            processed_df['description'] += processed_df.apply(
                lambda row: f". {col_name}: {row[column]}" if pd.notna(row[column]) and str(row[column]).strip() != '' else "",
                axis=1
            )

    # Clean up the description
    processed_df['description'] = processed_df['description'].str.replace('..', '.').str.strip()

    # Keep the combined_text field for backward compatibility
    processed_df['combined_text'] = processed_df['description']

    return processed_df

def create_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert DataFrame to a list of LangChain Document objects for vector store

    Args:
        df (pd.DataFrame): Preprocessed DataFrame

    Returns:
        List[Document]: List of LangChain Document objects
    """
    documents = []

    for _, row in df.iterrows():
        # Create metadata dictionary
        metadata = {}

        # Define essential fields for metadata (used for reference)

        # Add all available fields to metadata
        for column in df.columns:
            if column not in ['combined_text', 'description'] and not pd.isna(row[column]) and row[column] != '':
                metadata[column.lower().replace(' ', '_')] = row[column]

        # Ensure URL is always in metadata
        if 'URL' in row and not pd.isna(row['URL']) and row['URL'] != '':
            metadata['image_url'] = row['URL']

        # Create LangChain Document
        doc = Document(
            page_content=row['description'],
            metadata=metadata
        )
        documents.append(doc)

    return documents

def process_data() -> List[Document]:
    """
    Load, preprocess, and convert data to documents

    Returns:
        List[Document]: List of LangChain Document objects
    """
    ensure_cache_dir()
    df = load_csv_data()
    processed_df = preprocess_data(df)
    documents = create_documents(processed_df)
    return documents

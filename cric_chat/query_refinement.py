# Query refinement module for the Cricket Image Chatbot
# Provides functions to refine queries when no results are found

import re
import nltk
import os
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import config

def download_nltk_resources():
    # Download all required NLTK resources if they're not already available
    resources = [
        'punkt',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt'
                          else f'corpora/{resource}' if resource in ['wordnet', 'omw-1.4']
                          else f'taggers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)

# Download resources when module is imported
download_nltk_resources()

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def load_reference_data():
    # Load reference data from CSV files to build comprehensive entity variations
    # Returns dictionary containing entity variations
    try:
        # Define paths to reference data files
        data_files = {
            "players": os.path.join(config.DATA_DIR, "Players.csv"),
            "actions": os.path.join(config.DATA_DIR, "Action.csv"),
            "events": os.path.join(config.DATA_DIR, "Event.csv"),
            "moods": os.path.join(config.DATA_DIR, "Mood.csv"),
            "sublocations": os.path.join(config.DATA_DIR, "Sublocation.csv")
        }

        # Create entity variations dictionary
        entity_variations = {
            "players": {},
            "actions": {},
            "events": {},
            "moods": {},
            "sublocations": {}
        }

        # Process each entity type
        entity_df_map = {
            "players": pd.read_csv(data_files["players"]),
            "actions": pd.read_csv(data_files["actions"]),
            "events": pd.read_csv(data_files["events"]),
            "moods": pd.read_csv(data_files["moods"]),
            "sublocations": pd.read_csv(data_files["sublocations"])
        }

        column_name_map = {
            "players": "Player Name",
            "actions": "action_name",
            "events": "event_name",
            "moods": "mood_name",
            "sublocations": "sublocation_name"
        }

        variation_function_map = {
            "players": generate_player_name_variations,
            "actions": generate_action_variations,
            "events": generate_event_variations,
            "moods": generate_mood_variations,
            "sublocations": generate_sublocation_variations
        }

        # Process all entity types
        for entity_type, df in entity_df_map.items():
            column_name = column_name_map[entity_type]
            variation_function = variation_function_map[entity_type]

            for _, row in df.iterrows():
                entity_name = row[column_name]
                variations = variation_function(entity_name)
                entity_variations[entity_type][entity_name.lower()] = variations

        return entity_variations
    except Exception as e:
        print(f"Error loading reference data: {e}")
        return {key: {} for key in ["players", "actions", "events", "moods", "sublocations"]}

# Functions to generate variations for different entity types
def generate_player_name_variations(player_name: str) -> List[str]:
    # Generate variations of a player name
    # Takes player name and returns list of name variations
    variations = []
    name = player_name.strip()

    # Add the original name and lowercase version
    variations.extend([name, name.lower()])

    # Add name without spaces
    if ' ' in name:
        variations.extend([name.replace(' ', ''), name.lower().replace(' ', '')])

    # Split the name into parts
    name_parts = name.split()

    if len(name_parts) > 1:
        # Add first and last name variations
        variations.extend([name_parts[0], name_parts[0].lower(), name_parts[-1], name_parts[-1].lower()])

        # Add initials + last name variations
        initials = ''.join([part[0] for part in name_parts[:-1]])
        period_initials = '.'.join([part[0] for part in name_parts[:-1]]) + '.'

        initial_variations = [
            f"{initials} {name_parts[-1]}",
            f"{initials}{name_parts[-1]}",
            f"{initials}.{name_parts[-1]}",
            f"{period_initials} {name_parts[-1]}",
            f"{period_initials}{name_parts[-1]}"
        ]

        variations.extend(initial_variations)
        variations.extend([v.lower() for v in initial_variations])

    # Special case handling for specific players
    special_cases = {
        "FAF DU PLESSIS": ["faf", "du plessis", "duplessis", "faf duplessis"],
        "MOEEN ALI": ["moen ali", "mo ali", "moeen", "moen"],
        "JP KING": ["j.p. king", "j p king", "j.p king", "jp", "king"],
        "STEPHEN FLEMING": ["fleming", "steve fleming", "stephen", "steve"]
    }

    if name.upper() in special_cases:
        variations.extend(special_cases[name.upper()])

    # Remove duplicates and return
    return list(set(variations))

def generate_action_variations(action_name: str) -> List[str]:
    # Generate variations of an action name
    # Takes action name and returns list of action variations
    variations = []
    name = action_name.strip()

    # Add basic variations
    variations.extend([
        name,
        name.lower(),
        stemmer.stem(name.lower()),
        lemmatizer.lemmatize(name.lower(), pos='v')  # Assume it's a verb
    ])

    # Add present continuous form (if applicable)
    if not name.lower().endswith('ing'):
        if name.lower().endswith('e'):
            variations.append(f"{name.lower()[:-1]}ing")
        else:
            variations.append(f"{name.lower()}ing")

    # Add specific variations based on the action
    action_specific_variations = {
        "bowling": ["bowl", "bowls", "bowled", "throw", "throwing", "pitch", "pitching"],
        "batting": ["bat", "bats", "batted", "hit", "hitting", "strike", "striking"],
        "fielding": ["field", "fields", "fielded", "catch", "catching", "stop", "stopping"],
        "celebrating": ["celebrate", "celebrates", "celebrated", "cheer", "cheering", "rejoice"],
        "wicketkeeping": ["keep", "keeper", "keeping", "wicket keeper", "wk", "behind stumps"],
        "training": ["train", "trains", "trained", "practice", "practicing", "drill", "drilling"],
        "sitting": ["sit", "sits", "seated", "resting", "rest"],
        "walking": ["walk", "walks", "walked", "stroll", "strolling"],
        "catching": ["catch", "catches", "caught", "take", "taking", "grab", "grabbing"],
        "resting": ["rest", "rests", "rested", "relax", "relaxing", "break"],
        "posing": ["pose", "poses", "posed", "stand", "standing", "photo"],
        "signing": ["sign", "signs", "signed", "autograph", "autographing"],
        "talking": ["talk", "talks", "talked", "speak", "speaking", "chat", "chatting"],
        "greeting": ["greet", "greets", "greeted", "welcome", "welcoming", "meet", "meeting"],
        "strategizing": ["strategize", "plan", "planning", "discuss", "discussing", "analyze"],
        "sprinting": ["sprint", "sprints", "sprinted", "run", "running", "dash", "dashing"],
        "running": ["run", "runs", "ran", "jog", "jogging", "sprint", "sprinting"],
        "jogging": ["jog", "jogs", "jogged", "run", "running", "trot", "trotting"],
        "stretching": ["stretch", "stretches", "stretched", "warm up", "warming up"],
        "appealing": ["appeal", "appeals", "appealed", "request", "requesting", "ask", "asking"],
        "travelling": ["travel", "travels", "travelled", "journey", "journeying", "trip", "tripping"],
        "standing": ["stand", "stands", "stood", "wait", "waiting", "pose", "posing"]
    }

    # Add action-specific variations if available
    if name.lower() in action_specific_variations:
        variations.extend(action_specific_variations[name.lower()])

    # Remove duplicates and return
    return list(set(variations))

def generate_event_variations(event_name: str) -> List[str]:
    # Generate variations of an event name
    # Takes event name and returns list of event variations
    variations = [event_name.strip(), event_name.strip().lower()]

    # Add specific variations based on the event
    event_specific_variations = {
        "practice": ["training", "net session", "nets", "drill", "workout", "preparation",
                    "practice session", "training session", "warm-up", "warm up"],
        "match": ["game", "fixture", "contest", "tournament", "series", "competition",
                 "play", "playing", "cricket match", "t20", "t20 match"],
        "promotional event": ["promotion", "marketing", "advertisement", "commercial",
                             "sponsorship", "brand event", "promo", "marketing event"],
        "fan engagement": ["fan meet", "meet and greet", "fan interaction", "autograph session",
                          "fan event", "meet fans", "fan meeting", "supporter event"],
        "press meet": ["press conference", "media briefing", "interview", "media interaction",
                      "press", "media", "presser", "news conference", "media event"]
    }

    # Add event-specific variations if available
    if event_name.strip().lower() in event_specific_variations:
        variations.extend(event_specific_variations[event_name.strip().lower()])

    # Remove duplicates and return
    return list(set(variations))

def generate_mood_variations(mood_name: str) -> List[str]:
    # Generate variations of a mood name
    # Takes mood name and returns list of mood variations
    variations = [mood_name.strip(), mood_name.strip().lower()]

    # Add specific variations based on the mood
    mood_specific_variations = {
        "casual": ["relaxed", "informal", "laid-back", "easygoing", "chill", "comfortable",
                  "normal", "everyday", "regular", "standard"],
        "celebratory": ["celebrating", "happy", "joyful", "excited", "jubilant", "elated",
                       "thrilled", "ecstatic", "festive", "triumphant", "victorious"],
        "formal": ["official", "serious", "professional", "business-like", "proper",
                  "dignified", "ceremonial", "solemn", "composed", "reserved"]
    }

    # Add mood-specific variations if available
    if mood_name.strip().lower() in mood_specific_variations:
        variations.extend(mood_specific_variations[mood_name.strip().lower()])

    # Remove duplicates and return
    return list(set(variations))

def generate_sublocation_variations(sublocation_name: str) -> List[str]:
    # Generate variations of a sublocation name
    # Takes sublocation name and returns list of sublocation variations
    variations = [sublocation_name.strip(), sublocation_name.strip().lower()]

    # Add specific variations based on the sublocation
    sublocation_specific_variations = {
        "practice nets": ["nets", "net practice", "batting nets", "bowling nets", "training nets",
                         "practice area", "net area", "practice facility", "training facility"],
        "stadium": ["ground", "field", "venue", "arena", "cricket ground", "cricket stadium",
                   "sports ground", "pitch", "playing field", "cricket field"],
        "field": ["ground", "outfield", "playing area", "pitch", "cricket field", "playing field",
                 "grass", "turf", "playing surface"],
        "hotel": ["accommodation", "lodging", "team hotel", "residence", "place of stay",
                 "living quarters", "team accommodation"],
        "stage": ["platform", "podium", "dais", "presentation area", "award stage",
                 "ceremony stage", "event stage"],
        "locker room": ["dressing room", "change room", "team room", "players' room",
                       "changing area", "team area", "players' area"],
        "restaurant": ["dining area", "cafe", "eatery", "dining place", "food court",
                      "dining hall", "cafeteria", "food place"],
        "airport": ["terminal", "air terminal", "airfield", "departure area", "arrival area",
                   "travel hub", "transit area"]
    }

    # Add sublocation-specific variations if available
    if sublocation_name.strip().lower() in sublocation_specific_variations:
        variations.extend(sublocation_specific_variations[sublocation_name.strip().lower()])

    # Remove duplicates and return
    return list(set(variations))

# Define cricket-specific synonyms
CRICKET_SYNONYMS = {
    # Player positions and roles
    "batsman": ["batter", "hitter", "striker", "batman"],
    "bowler": ["pitcher", "thrower", "pacer", "spinner"],
    "wicketkeeper": ["keeper", "wk", "wicket keeper", "gloveman"],
    "fielder": ["catcher", "outfielder", "infielder"],
    "all-rounder": ["all rounder", "allrounder"],

    # Actions
    "batting": ["hitting", "striking", "playing", "shot-making"],
    "bowling": ["throwing", "pitching", "delivering", "tossing"],
    "fielding": ["catching", "stopping", "diving", "intercepting"],
    "celebrating": ["cheering", "rejoicing", "jubilant", "happy", "excited"],

    # Events
    "match": ["game", "fixture", "contest", "tournament", "series"],
    "practice": ["training", "net session", "drill", "workout", "preparation"],
    "press meet": ["press conference", "media briefing", "interview", "media interaction"],
    "promotional event": ["marketing event", "advertisement", "commercial", "sponsorship"],

    # Equipment
    "bat": ["willow", "blade"],
    "ball": ["cherry", "leather", "kookaburra", "duke"],
    "stumps": ["wicket", "timber", "woodwork"],
    "helmet": ["headgear", "head protection"],
    "gloves": ["mitts", "hand protection"],

    # Locations
    "stadium": ["ground", "field", "venue", "arena"],
    "pitch": ["wicket", "strip", "track", "surface"],
    "boundary": ["rope", "fence", "perimeter"],
    "nets": ["practice area", "training facility"],

    # Team-specific
    "joburg super kings": ["jsk", "johannesburg super kings", "super kings"],
    "team": ["squad", "side", "outfit", "eleven", "lineup"]
}

def get_synonyms(word: str) -> List[str]:
    # Get synonyms for a word using WordNet and cricket-specific synonyms
    # Takes a word and returns list of synonyms
    word_lower = word.lower()

    # Check cricket-specific synonyms first
    for key, values in CRICKET_SYNONYMS.items():
        if word_lower == key:
            return values
        if word_lower in values:
            return [key] + [v for v in values if v != word_lower]

    # Get synonyms from WordNet
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym not in synonyms:
                synonyms.append(synonym)

    return synonyms

def stem_word(word: str) -> str:
    # Stem a word using Porter stemmer
    return stemmer.stem(word)

def get_word_stems(text: str) -> List[str]:
    # Get stems of words in a text
    words = word_tokenize(text.lower())
    return [stem_word(word) for word in words if word.isalnum()]

def generate_refined_queries(query: str) -> List[str]:
    # Generate refined queries based on the original query
    # Takes original query and returns list of refined queries
    refined_queries = [query]  # Add the original query
    words = word_tokenize(query.lower())

    # Get POS tags
    try:
        pos_tags = nltk.pos_tag(words)
    except LookupError:
        # Fallback: assign 'NN' (noun) tag to all words if POS tagging fails
        pos_tags = [(word, 'NN') for word in words]

        # Try to download the correct resource
        try:
            nltk.download('averaged_perceptron_tagger')
            # Try again with the downloaded resource
            pos_tags = nltk.pos_tag(words)
        except Exception:
            pass

    # Generate queries with synonyms
    for i, (word, pos) in enumerate(pos_tags):
        # Only consider nouns, verbs, and adjectives
        if pos.startswith('N') or pos.startswith('V') or pos.startswith('J'):
            synonyms = get_synonyms(word)
            for synonym in synonyms:
                # Create a new query by replacing the word with its synonym
                new_words = words.copy()
                new_words[i] = synonym
                new_query = ' '.join(new_words)
                if new_query not in refined_queries:
                    refined_queries.append(new_query)

    # Generate queries with stemming
    stems = get_word_stems(query)
    stem_query = ' '.join(stems)
    if stem_query not in refined_queries:
        refined_queries.append(stem_query)

    # Generate queries with both synonyms and stemming
    for i, (word, pos) in enumerate(pos_tags):
        if pos.startswith('N') or pos.startswith('V') or pos.startswith('J'):
            synonyms = get_synonyms(word)
            for synonym in synonyms:
                new_words = words.copy()
                new_words[i] = synonym
                new_stems = get_word_stems(' '.join(new_words))
                new_stem_query = ' '.join(new_stems)
                if new_stem_query not in refined_queries:
                    refined_queries.append(new_stem_query)

    return refined_queries

def correct_spelling(query: str) -> str:
    # Correct spelling in a query using a basic approach
    # Takes query and returns corrected query
    words = word_tokenize(query.lower())

    # Define common cricket terms that might be misspelled
    cricket_terms = {
        # Common misspellings -> correct spelling
        "batsman": ["batsmen", "batsmans", "batsman", "batsmen", "batters"],
        "bowler": ["bowelers", "bowlers", "bowlar", "bowlar"],
        "wicketkeeper": ["wicket keeper", "wicket-keeper", "keeper", "wk", "wkt keeper"],
        "fielder": ["fielders", "felder", "feilder"],
        "batting": ["bating", "battting", "bating"],
        "bowling": ["boweling", "bowlling", "bowlng"],
        "fielding": ["feildng", "fieldin", "feilding"],
        "celebrating": ["celebratng", "celebratig", "celabrating"],
        "practice": ["practise", "practce", "practis"],
        "match": ["mach", "matche", "mathc"],
        "joburg": ["joberg", "johburg", "joburg", "johannesburg"],
        "kings": ["king", "kigns", "kngs"],
        "stadium": ["stadum", "stedium", "statium"],
        "cricket": ["criket", "cricet", "crickt"],
        "player": ["playr", "pleyer", "plyer"],
        "team": ["tem", "teem", "taem"],
        "press": ["pres", "prss", "presss"],
        "conference": ["conferance", "confrence", "conferrence"],
        "interview": ["intervew", "intervue", "interveiw"],
        "promotional": ["promotinal", "promotonal", "promtional"],
        "event": ["evnt", "eventt", "evant"]
    }

    # Correct each word if it's a misspelling of a cricket term
    corrected_words = []
    for word in words:
        corrected = word
        for correct_term, misspellings in cricket_terms.items():
            if word in misspellings:
                corrected = correct_term
                break
        corrected_words.append(corrected)

    return ' '.join(corrected_words)

# Load entity variations when module is imported
entity_variations = load_reference_data()

def refine_query(query: str) -> List[str]:
    # Refine a query using multiple techniques including entity-specific variations
    # Takes original query and returns list of refined queries
    refined_queries = [query]  # Add the original query

    # 1. Correct spelling
    corrected_query = correct_spelling(query)
    if corrected_query != query:
        refined_queries.append(corrected_query)

    # 2. Generate queries with synonyms and stemming
    refined_queries.extend(generate_refined_queries(query))

    # 3. Try entity-specific variations
    refined_queries.extend(generate_entity_specific_queries(query))

    # 4. Try removing stop words
    stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after', 'between', 'under', 'during', 'without', 'of']
    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word.lower() not in stop_words]
    if filtered_words:
        filtered_query = ' '.join(filtered_words)
        if filtered_query not in refined_queries:
            refined_queries.append(filtered_query)

    # 5. Try keyword extraction, reordering, and lemmatization
    try:
        pos_tags = nltk.pos_tag(words)

        # Extract nouns and verbs (most important for search)
        keywords = [word for word, tag in pos_tags if tag.startswith('NN') or tag.startswith('VB')]
        if len(keywords) >= 2:
            # Try different orderings of keywords
            for i in range(len(keywords)):
                reordered = keywords[i:] + keywords[:i]
                reordered_query = ' '.join(reordered)
                if reordered_query not in refined_queries:
                    refined_queries.append(reordered_query)

        # Try lemmatization for verbs in the query
        lemmatized_pairs = []  # Store (original, lemmatized) pairs
        for word, tag in pos_tags:
            if tag.startswith('VB'):  # If it's a verb
                lemmatized_word = lemmatizer.lemmatize(word, pos='v')
                if lemmatized_word != word:
                    lemmatized_pairs.append((word, lemmatized_word))

        # Replace verbs with their lemmatized forms
        if lemmatized_pairs:
            lemma_query = query.lower()
            for original, lemmatized in lemmatized_pairs:
                lemma_query = lemma_query.replace(original, lemmatized)
            if lemma_query not in refined_queries:
                refined_queries.append(lemma_query)
    except Exception as e:
        print(f"Error in keyword extraction or lemmatization: {e}")

    # 6. Try adding cricket-specific context terms
    context_terms = ["cricket", "player", "match", "joburg super kings", "jsk"]
    for term in context_terms:
        if term not in query.lower():
            contextual_query = f"{query} {term}"
            if contextual_query not in refined_queries:
                refined_queries.append(contextual_query)

    # Remove duplicates and empty queries
    refined_queries = [q for q in refined_queries if q.strip()]
    return list(dict.fromkeys(refined_queries))

def generate_entity_specific_queries(query: str) -> List[str]:
    # Generate refined queries using entity-specific variations (players, actions, events, moods, sublocations)
    # Takes original query and returns list of refined queries with entity-specific variations
    refined_queries = []
    query_lower = query.lower()

    # Process all entity types in a single loop
    entity_types = ["players", "actions", "events", "moods", "sublocations"]

    for entity_type in entity_types:
        for entity_name, variations in entity_variations[entity_type].items():
            for variation in variations:
                if variation in query_lower:
                    # Replace the variation with other variations
                    for alt_variation in variations:
                        if alt_variation != variation:
                            refined_query = query_lower.replace(variation, alt_variation)
                            refined_queries.append(refined_query)

    # Handle special case for multiple player queries
    multi_player_indicators = ["and", "&", ",", "with", "together", "same frame", "single frame"]
    if any(indicator in query_lower for indicator in multi_player_indicators):
        # Identify player names in the query
        identified_players = []
        for player_name, variations in entity_variations["players"].items():
            for variation in variations:
                if variation in query_lower:
                    identified_players.append((player_name, variation))

        if identified_players:
            # Try different ways to combine players
            connectors = [" and ", " & ", ", ", " with ", " alongside ", " together with "]

            # Try different variations of the identified players
            for player_name, variation in identified_players:
                all_variations = entity_variations["players"][player_name]

                # Replace the current variation with other variations
                for alt_variation in all_variations:
                    if alt_variation != variation:
                        refined_query = query_lower.replace(variation, alt_variation)
                        refined_queries.append(refined_query)

                        # Also try with different connectors
                        existing_connectors = [" and ", " & ", ", ", " with ", " alongside "]
                        for connector in connectors:
                            for existing_connector in existing_connectors:
                                if existing_connector in query_lower:
                                    connector_refined = refined_query.replace(existing_connector, connector)
                                    refined_queries.append(connector_refined)

            # Add special terms if not already present
            special_term_groups = [
                ["together", "in the same frame", "in a single frame", "in one frame"],
                ["with multiple faces", "with at least 2 faces", "group photo"]
            ]

            for term_group in special_term_groups:
                if not any(term in query_lower for term in term_group):
                    for term in term_group:
                        refined_queries.append(f"{query_lower} {term}")

    return refined_queries

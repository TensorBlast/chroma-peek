import chromadb 
import pandas as pd
import numpy as np

class ChromaPeek:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)

    ## function that returs all collection's name
    def get_collections(self):
        collections = []

        for i in self.client.list_collections():
            collections.append(i.name)
        
        return collections
    
    ## private method to flatten metadata into separate columns
    def _flatten_metadata(self, metadatas):
        """
        Flatten metadata dictionaries into separate columns.
        Handles nested dictionaries and varying metadata fields across documents.
        """
        if not metadatas:
            return {}
        
        # Collect all possible metadata keys from all documents
        all_keys = set()
        for metadata in metadatas:
            if metadata:
                all_keys.update(self._get_nested_keys(metadata))
        
        # Create columns for each metadata key
        flattened = {}
        for key in sorted(all_keys):  # Sort for consistent column order
            column_name = f"metadata_{key}"
            flattened[column_name] = []
            
            for metadata in metadatas:
                nested_dict = self._get_nested_dict(metadata)
                if metadata and key in nested_dict:
                    flattened[column_name].append(nested_dict.get(key))
                else:
                    flattened[column_name].append(None)
        
        return flattened
    
    def _get_nested_keys(self, d, prefix=''):
        """Recursively get all keys from nested dictionary."""
        keys = set()
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                keys.add(new_key)
                if isinstance(v, dict):
                    keys.update(self._get_nested_keys(v, new_key))
        return keys
    
    def _get_nested_dict(self, d):
        """Flatten nested dictionary into dot-notation keys."""
        result = {}
        
        def _flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        _flatten(v, new_key)
                    else:
                        result[new_key] = v
        
        _flatten(d)
        return result
    
    
    ## private method to convert data to dataframe format
    def _to_dataframe(self, data):
        """
        Convert Chroma data dictionary to pandas DataFrame format.
        Handles different array lengths, special formatting for embeddings, and metadata flattening.
        """
        df_data = {}
        max_length = 0
        
        # Find the maximum length among non-None arrays
        for key, value in data.items():
            if value is not None and isinstance(value, list):
                max_length = max(max_length, len(value))
        
        # Process each field to ensure consistent length
        for key, value in data.items():
            if key == 'embeddings' and value is not None:
                # Convert embeddings to readable format
                df_data[key] = [f"Vector({len(emb)} dims)" if emb is not None else None for emb in value]
            elif key == 'metadatas' and value is not None:
                # Flatten metadata into separate columns
                flattened_metadata = self._flatten_metadata(value)
                df_data.update(flattened_metadata)
            elif key != 'metadatas':  # Skip metadatas as we handle it separately above
                if value is None:
                    df_data[key] = [None] * max_length
                elif isinstance(value, list):
                    if len(value) == max_length:
                        df_data[key] = value
                    elif len(value) < max_length:
                        # Pad shorter arrays with None
                        df_data[key] = value + [None] * (max_length - len(value))
                    else:
                        # Truncate longer arrays (shouldn't happen in normal cases)
                        df_data[key] = value[:max_length]
                else:
                    # For non-list values, repeat the value for all rows
                    df_data[key] = [value] * max_length
        
        return pd.DataFrame(df_data)
    
    ## function to return documents/ data inside the collection
    def get_collection_data(self, collection_name, dataframe=False, include=None):
        if include is None:
            include = ['documents', 'metadatas']  # Default: exclude embeddings
        data = self.client.get_collection(name=collection_name).get(include=include)
        if dataframe:
            return self._to_dataframe(data)
        return data
    
    ## function to query the selected collection
    def query(self, query_str, collection_name, k=3, dataframe=False, include=None):
        if include is None:
            include = ['documents', 'metadatas']  # Default: exclude embeddings
        collection = self.client.get_collection(collection_name)
        res = collection.query(
            query_texts=[query_str], n_results=min(k, len(collection.get())),
            include=include
        )
        out = {}
        for key, value in res.items():
            if value:
                out[key] = value[0]
            else:
                out[key] = value
        if dataframe:
            return self._to_dataframe(out)
        return out
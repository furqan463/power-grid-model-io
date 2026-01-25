# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
PSS/E® V33 ".raw" File Store
"""


import pandas as pd
import numpy as np
import structlog
from typing import Optional, Dict, List, Tuple, Union, TextIO, Any, Type
from pathlib import Path
from abc import ABC

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import LazyDataFrame, PsseData

class PsseDataStore(BaseDataStore[PsseData]):
    """
    PSS/E .raw File Store
    
    Parses PSS/E raw files into TabularData.
    Since .raw files do not have headers, this class uses a predefined SCHEMA
    to map data columns to names and units.
    """
    
    __slots__ = ("_source", "_encoding", "_psse_version")

    # Define the Schema: Key = Section Name, Value = List of (ColumnName, Unit, Dtype)
    # This acts as the "Header" definitions that are missing in .raw files.
    # TODO: Length units
    _SCHEMA: Dict[str, List[Tuple[str, str, Type]]] = {
        "bus": [
            ("I", "", int), ("NAME", "", str), ("BASKV", "kV", float), ("IDE", "", int), 
            ("AREA", "", int), ("ZONE", "", int), ("OWNER", "", int), ("VM", "pu", float), 
            ("VA", "deg", float), ("NVHI", "pu", float), ("NVLO", "pu", float), 
            ("EVHI", "pu", float), ("EVLO", "pu", float)
        ],
        "load": [
            ("I", "", int), ("ID", "", str), ("STATUS", "", int), ("AREA", "", int), 
            ("ZONE", "", int), ("PL", "MW", float), ("QL", "Mvar", float), 
            ("IP", "MW", float), ("IQ", "Mvar", float), ("YP", "MW", float), 
            ("YQ", "Mvar", float), ("OWNER", "", int), ("SCALE", "", int),
            ("INTRPT", "", int)
        ],
        "fixed_shunt": [
            ("I", "", int), ("ID", "", str), ("STATUS", "", int), ("GL", "MW", float),
            ("BL", "Mvar", float)
        ],
        "generator": [
            ("I", "", int), ("ID", "", str), ("PG", "MW", float), ("QG", "Mvar", float),
            ("QT", "Mvar", float), ("QB", "Mvar", float), ("VS", "pu", float), 
            ("IREG", "", int), ("MBASE", "MVA", float), ("ZR", "pu", float), 
            ("ZX", "pu", float), ("RT", "pu", float), ("XT", "pu", float), 
            ("GTAP", "pu", float), ("STAT", "", int), ("RMPCT", "", float), 
            ("PT", "MW", float), ("PB", "MW", float), ("O1", "", int), ("F1", "", float)
        ],
        "branch": [
            ("I", "", int), ("J", "", int), ("CKT", "", str), ("R", "pu", float),
            ("X", "pu", float), ("B", "pu", float), ("RATEA", "MVA", float),
            ("RATEB", "MVA", float), ("RATEC", "MVA", float), ("GI", "pu", float),
            ("BI", "pu", float), ("GJ", "pu", float), ("BJ", "pu", float),
            ("ST", "", int), ("MET", "", int), ("LEN", "km", float), ("O1", "", int),
            ("F1", "", float)
        ],
        "transformer": [
            ("I", "", int), ("J", "", int), ("K", "", int), ("CKT", "", str),
            ("CW", "", int), ("CZ", "", int), ("CM", "", int), ("MAG1", "pu", float),
            ("MAG2", "pu", float), ("NMETR", "", int), ("NAME", "", str), ("STAT", "", int),
            ("O1", "", int), ("F1", "", float), ("VECGRP", "", str), ("R1-2", "pu", float),
            ("X1-2", "pu", float), ("SBASE1-2", "pu", float), ("WINDV1", "pu", float),
            ("NOMV1", "kV", float), ("ANG1", "deg", float), ("RATA1", "MVA", float),
            ("RATB1", "MVA", float), ("RATC1", "MVA", float), ("COD1", "", int),
            ("CONT1", "", int), ("RMA1", "pu", float), ("RMI1", "pu", float),
            ("VMA1", "pu", float), ("VMI1", "pu", float), ("NTP1", "", int),
            ("TAB1", "", int), ("CR1", "pu", float), ("CX1", "pu", float),
            ("CNXA1", "deg", float), ("WINDV2", "pu", float), ("NOMV2", "kV", float)
        ],
        "three_winding_transformer": [
            ("I", "", int), ("J", "", int), ("K", "", int), ("CKT", "", str),
            ("CW", "", int), ("CZ", "", int), ("CM", "", int), ("MAG1", "pu", float),
            ("MAG2", "pu", float), ("NMETR", "", int), ("NAME", "", str), ("STAT", "", int),
            ("O1", "", int), ("F1", "", float), ("VECGRP", "", str), ("R1-2", "pu", float),
            ("X1-2", "pu", float), ("SBASE1-2", "pu", float), ("R2-3", "pu", float),
            ("X2-3", "pu", float), ("SBASE2-3", "pu", float), ("R3-1", "pu", float),
            ("X3-1", "pu", float), ("SBASE3-1", "pu", float), ("VMSTAR", "pu", float),
            ("ANSTAR", "deg", float), ("WINDV1", "pu", float),
            ("NOMV1", "kV", float), ("ANG1", "deg", float), ("RATA1", "MVA", float),
            ("RATB1", "MVA", float), ("RATC", "MVA", float), ("COD1", "", int),
            ("CONT1", "", int), ("RMA1", "pu", float), ("RMI1", "pu", float),
            ("VMA1", "pu", float), ("VMI1", "pu", float), ("NTP1", "", int),
            ("TAB1", "", int), ("CR1", "pu", float), ("CX1", "pu", float),
            ("CNXA1", "deg", float), ("WINDV2", "pu", float), ("NOMV2", "kV", float),
            ("ANG2", "deg", float), ("RATA2", "MVA", float), ("RATB2", "MVA", float),
            ("RATC2", "MVA", float), ("COD2", "", int), ("CONT2", "", int),
            ("RMA2", "pu", float), ("RMI2", "pu", float),
            ("VMA2", "pu", float), ("VMI2", "pu", float), ("NTP2", "", int),
            ("TAB2", "", int), ("CR2", "pu", float), ("CX2", "pu", float),
            ("CNXA2", "deg", float), ("WINDV3", "pu", float), ("NOMV3", "kV", float),
            ("ANG3", "deg", float), ("RATA3", "MVA", float), ("RATB3", "MVA", float),
            ("RATC3", "MVA", float), ("COD3", "", int), ("CONT3", "", int),
            ("RMA3", "pu", float), ("RMI3", "pu", float),
            ("VMA3", "pu", float), ("VMI3", "pu", float), ("NTP3", "", int),
            ("TAB3", "", int), ("CR3", "pu", float), ("CX3", "pu", float),
            ("CNXA3", "deg", float)
        ]
        # Add other sections (branches, transformers) as needed...
    }

    # TODO: Support file-like object (Input Stream)
    def __init__(
        self, 
        source: Union[str, Path],
        encoding: str = "latin-1"
    ):
        """
        Initialize the store.
        
        Args:
            source: File path (str/Path)
            encoding: Encoding of the text file (default 'latin-1' for PSS/E)
        """
        super().__init__()
        self._source = source
        self._encoding = encoding
        self._psse_version = 33 # Default fallback, usually detected in first line

    def load(self) -> PsseData:
        """
        Load the PSS/E raw file data into PsseData.
        """
        data_dict: Dict[str, pd.DataFrame] = {}

        # 1. Open Stream
        if isinstance(self._source, (str, Path)):
            self._log.info(f"Loading PSS/E file: {self._source}")
            with open(self._source, "r", encoding=self._encoding) as f:
                raw_sections = self._parse_stream(f)
        else:
            raise TypeError(f"Invalid source type: {type(self._source)}")

        print(raw_sections)
        # 2. Convert Raw Lists to DataFrames using Schema
        for section_name, rows in raw_sections.items():
            if section_name in self._SCHEMA:
                self._log.debug(f"Processing section: {section_name}, rows: {len(rows)}")
                df = self._create_dataframe(rows, self._SCHEMA[section_name])
                data_dict[section_name] = df
            else:
                self._log.warning(f"Skipping unknown section: {section_name}")

        s_base = 100
        f = 50
        return PsseData(s_base=s_base, frequency=f, **data_dict)

    def save(self, data: PsseData) -> None:
        raise NotImplementedError("Saving to .raw format is not yet supported.")

    def _parse_stream(self, stream: TextIO) -> Dict[str, List[List[Any]]]:
        """
        Reads the text stream and splits it into sections (Buses, Loads, etc.)
        based on PSS/E "0 / END OF DATA" markers.
        """
        sections: Dict[str, List[List[Any]]] = {}
        
        # PSS/E Section Order (simplified for version 33)
        # We need a map because the file doesn't say "BUSES", it just assumes order.
        # TODO: This must come from the schema.
        section_order = [
            "bus", "load", "fixed_shunt", "generator", "branch"
        ]
        
        current_section_idx = 0
        current_rows = []
        
        # Helper to push current data to dict
        def push_section():
            nonlocal current_section_idx, current_rows
            if current_section_idx < len(section_order):
                sec_name = section_order[current_section_idx]
                sections[sec_name] = current_rows
            current_rows = []
            current_section_idx += 1

        # Skip the first 3 lines (Case ID lines)
        header_lines = [stream.readline() for _ in range(3)]
        # You might parse version from header_lines[0] here
        
        for line in stream:
            line = line.strip()
            if not line: 
                continue
                
            # Check for Section End Marker
            # Usually starts with '0' or 'Q' in first column
            if line.startswith('0') or line.startswith('Q'):
                push_section()
                continue
            
            # Parse line CSV style
            # Note: PSS/E can use commas or spaces. This is a naive splitter.
            # Production code might need a more robust regex splitter for quoted strings.
            parts = [x.strip("'").strip() for x in line.split(',')]
            current_rows.append(parts)

        return sections

    def _create_dataframe(self, rows: List[List[Any]], schema: List[Tuple[str, str, Type]]) -> pd.DataFrame:
        """
        Converts a list of raw string rows into a typed DataFrame with MultiIndex columns (Name, Unit).
        """
        if not rows:
            return pd.DataFrame()

        # 1. Prepare Columns (MultiIndex: Name, Unit)
        col_names = [(col[0], col[1]) for col in schema]
        multi_index = pd.MultiIndex.from_tuples(col_names)
        # 2. Create DataFrame
        # We assume rows match schema length. If .raw file has extra cols, we slice.
        # If .raw file has fewer cols, we pad (handled by pd.DataFrame usually or requires checks).
        df = pd.DataFrame(rows)
        
        # Resize if necessary (drop extra columns in file or handle mismatch)
        if df.shape[1] > len(schema):
             df = df.iloc[:, :len(schema)]
        
        # Set Columns
        df.columns = multi_index

        # 3. Apply Types
        # This is critical for TabularData to work with math later
        for col_def in schema:
            col_name, _, dtype = col_def
            try:
                # We access by Level 0 (Name)
                # Coerce errors=coerce turns invalid parsing to NaN
                if dtype == float:
                    df[col_name] = df[col_name].apply(pd.to_numeric, errors='coerce')
                elif dtype == int:
                    df[col_name] = df[col_name].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                elif dtype == str:
                    df[col_name] = df[col_name].astype(str)
            except KeyError:
                pass # Column might be missing if file was short
        return df
    
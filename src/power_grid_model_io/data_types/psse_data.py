# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
The PsseData class is a wrapper around TabularData,
whith specific structure of PSS/E elements
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from power_grid_model_io.data_types import LazyDataFrame, TabularData

class PsseData(TabularData):
    """
    Specific data container for PSS/E networks.
    Inherits from TabularData to provide semantic access and power-system 
    specific metadata.
    """

    def __init__(self, s_base: float = 100.0, frequency: float = 50.0, **tables: pd.DataFrame | np.ndarray | LazyDataFrame):
        """
        Initialize with PSS/E specific header information.
        
        Args:
            s_base: The system MVA base (S_base).
            frequency: The system frequency (Hz).
        """
        super().__init__(**tables)
        self.s_base = s_base
        self.frequency = frequency
        
        # Cache for fast ID lookups
        self._bus_index_cache: Optional[Dict[int, int]] = None

    def get(self, table_name: str, default: pd.DataFrame = None) -> pd.DataFrame:
        """
        Safely retrieve a table by name.
        """
        if table_name in self:
            return self[table_name]
        return default

    @property
    def bus(self) -> pd.DataFrame:
        return self.get("bus", pd.DataFrame())
    
    @property
    def load(self) -> pd.DataFrame:
        return self.get("load", pd.DataFrame())
    
    @property
    def fixed_shunt(self) -> pd.DataFrame:
        return self.get("fixed_shunt", pd.DataFrame())
    
    @property
    def generator(self) -> pd.DataFrame:
        return self.get("generator", pd.DataFrame())

    @property
    def branch(self) -> pd.DataFrame:
        return self.get("branch", pd.DataFrame())
    
    @property
    def transformer(self) -> pd.DataFrame:
        return self.get("transformer", pd.DataFrame())
    
    @property
    def three_winding_transformer(self) -> pd.DataFrame:
        return self.get("three_winding_transformer", pd.DataFrame())


    # --- Helper Methods ---

    def get_bus_index(self, bus_id: int) -> int:
        """
        Quickly find the DataFrame row index for a given PSS/E Bus Number.
        """
        if self._bus_index_cache is None:
            # We assume 'I' is the column name for Bus Number in the MultiIndex
            # Using level=0 to find the 'I' field regardless of the unit name.
            self._bus_index_cache = pd.Series(
                self.buses.index.values, 
                index=self.buses['I'].iloc[:, 0]
            ).to_dict()
        
        return self._bus_index_cache[bus_id]

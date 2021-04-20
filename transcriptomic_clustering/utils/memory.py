import os
from typing import Optional
from dataclasses import dataclass
import math

import psutil
import scanpy as sc


@dataclass
class Memory:
    """
    Object to hold transcriptomic clustering environment memory settings
    
    -----
    memory_limit_GB: memory in GB allowed to be used, -1 for no limit
    allow_chunking: whether to allow functions to load and process data in chunks
                    (slower but memory-friendly)
    """
    memory_limit_GB: int = -1
    allow_chunking: bool = False


    def set_memory_limit(self,
            GB: Optional[int]=None,
            percent_available: Optional[float]=None): 
        """
        sets memory limit for transcriptomic clustering functions

        Parameters:
        -----------
        GB: amount of memory in GB
        percent_available: percent of available memory

        """
        if GB and percent_available:
            raise ValueError("please pass GB or percent_available, not both")
        elif GB:
            self.memory_limit_GB = GB
        elif percent_available:
            if percent_available < 0 or percent_available > 100:
                raise ValueError('percent available must be between 0 and 100')
            self.memory_limit_GB = (self.get_available_memory_GB() * percent_available / 100)
        else:
            raise ValueError("please provide either percent_available or GB")


    def remove_memory_limit(self):
        self.memory_limit_GB = -1


    def get_available_memory_GB(self):
        """
        Returns available memory or memory limit, which ever is less
        """
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        if (self.memory_limit_GB == -1):
            return available_memory

        working_limit = ( # memory limit minus memory already used by this process
            self.memory_limit_GB - psutil.Process().memory_info().rss / (1024 ** 3)
        )
        if available_memory < working_limit:
            return available_memory
        else:
            return working_limit
    

    def estimate_n_chunks(self, process_memory, output_memory: Optional[float]=None, percent_available: Optional[float]=100):
        """
        Estimates appropriate number of chuncks based on memory need for total processing

        Parameters
        ----------
        process_memory: amount of memory in GB function is expected to need to process entire data
        output_memory: amount of memory that the function outputs will take
        percent_available: amount of available memory that can be spent on this function (default 50%)

        Returns
        -------
        Estimate of number of chunks to use. If output memory > available memory, returns -1 

        """
        available_memory = self.get_available_memory_GB() * (percent_available / 100)
        if output_memory:                
            available_memory -= output_memory
            if available_memory < 0:
                return -1 # not enough memory to even store the outputs

        nchunks = math.ceil(process_memory / available_memory)
        return nchunks
        

    def get_chunk_size(self, adata: sc.AnnData, n_chunks):
        return math.ceil(adata.n_obs / n_chunks)

memory = Memory()
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
    max_chunks = 5000


    def set_memory_limit(self,
            GB: Optional[int]=None,
            percent_current_available: Optional[float]=None):
        """
        sets memory limit for transcriptomic clustering functions

        Parameters:
        -----------
        GB: amount of memory in GB
        percent_current_available: percent of available memory

        """
        if GB and percent_current_available:
            raise ValueError("please pass GB or percent_current_available, not both")
        elif GB:
            self.memory_limit_GB = GB
        elif percent_current_available:
            if percent_current_available < 0 or percent_current_available > 100:
                raise ValueError('percent available must be between 0 and 100')
            self.memory_limit_GB = (self.get_available_system_memory_GB() * percent_current_available / 100)
        else:
            raise ValueError("please provide either percent_current_available or GB")


    def remove_memory_limit(self):
        self.memory_limit_GB = -1


    def get_available_system_memory_GB(self):
        """Returns available system memory in Gigabytes"""
        return psutil.virtual_memory().available / (1024 ** 3)


    def get_available_memory_GB(self):
        """
        Returns available memory or memory limit, which ever is less
        """
        available_system_memory = self.get_available_system_memory_GB()
        if (self.memory_limit_GB == -1):
            return available_system_memory

        available_process_memory_limited = ( # memory limit minus memory already used by this process
            self.memory_limit_GB - psutil.Process().memory_info().rss / (1024 ** 3)
        )
        return min(available_process_memory_limited, available_system_memory)
    

    def estimate_n_chunks(
            self,
            process_memory: float,
            output_memory: Optional[float]=None,
            percent_allowed: Optional[float]=None,
            process_name: Optional[str]=None):
        """
        Estimates appropriate number of chuncks based on memory need for total processing

        Parameters
        ----------
        process_memory: amount of memory in GB function is expected to need to process entire data
        output_memory: amount of memory that the function outputs will take
        percent_allowed: amount of available memory that can be spent on this function (default 50%)
        process_name: name of function/process to allocate (for error messages)

        Returns
        -------
        Estimate of number of chunks to use.
        Raises error if allow_chunking is False, if output_memory > available_memory, or n_chunks > max_n_chunks

        """
        if not percent_allowed:
            percent_allowed = 100
        if not process_name:
            process_name = "Operation"

        available_memory = self.get_available_memory_GB() * (percent_allowed / 100)
        if output_memory:                
            available_memory -= output_memory
        
        if available_memory < 0:
            nchunks = -1
        else:
            nchunks = math.ceil(process_memory / available_memory)

        if nchunks == 1:
            # always fine, no errors
            return nchunks
        elif nchunks == -1:
            raise MemoryError(
                f'{process_name} can not fit in memory!\n'
                f'available memory: {available_memory} GB, \n'
                f'output memory: {output_memory}'
            )
        elif self.allow_chunking and (nchunks > self.max_chunks):
            raise MemoryError(
                f'Chunking {process_name} requires too many chunks!'
                f'available memory: {available_memory} GB, '
                f'output memory: {output_memory}'
                f'n_chunks: {nchunks} > max_chunks {self.max_chunks}'
            )
        elif not self.allow_chunking:
            raise MemoryError(
                f'{process_name} could be done using chunking,'
                'set transcriptomic_clustering.memory.allow_chunking=True'
            )
        return nchunks
        

    def get_chunk_size(self, adata: sc.AnnData, n_chunks):
        if not adata.isbacked:
            raise MemoryError('Can not chunk in-memory AnnData')
        return math.ceil(adata.n_obs / n_chunks)

    def estimate_chunk_size(
            self,
            adata: sc.AnnData,
            process_memory: float,
            output_memory: Optional[float]=None,
            percent_allowed: Optional[float]=None,
            process_name: Optional[str]=None):
        """
        Estimates chunk size based on memory need for total processing

        Parameters
        ----------
        adata: AnnData object
        process_memory: amount of memory in GB function is expected to need to process entire data
        output_memory: amount of memory that the function outputs will take
        percent_allowed: amount of available memory that can be spent on this function (default 50%)
        process_name: name of function/process to allocate (for error messages)

        Returns
        -------
        Estimate of chunk size.
        Raises error if allow_chunking is False, if output_memory > available_memory, or n_chunks > max_n_chunks
        """
        n_chunks = self.estimate_n_chunks(
            process_memory,
            output_memory=output_memory,
            percent_allowed=percent_allowed,
            process_name=process_name
        )
        return self.get_chunk_size(adata, n_chunks)

memory = Memory()
import numpy
from typing import Type, List, Optional
from enum import Enum
import os
import logging
import shutil
import struct
import time
from tqdm import tqdm
from functools import lru_cache
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union, Any
import datasets
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"
VOCAB_SIZE = 99000


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """If torch distributed is initialized, log only on rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    # if torch.distributed.is_initialized():
    #     if torch.distributed.get_rank() == rank:
    #         logger.log(*args, **kwargs)
    # else:
    logger.log(*args, **kwargs)


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the index file
    """
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the data file
    """
    return path_prefix + ".bin"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class _IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[numpy.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: Type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(
        self,
        sequence_lengths: List[int],
        sequence_modes: Optional[List[int]],
        document_indices: List[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            sequence_modes (Optional[List[int]]): The mode of each sequences

            document_indices (List[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        sequence_lengths = numpy.array(sequence_lengths, dtype=numpy.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = numpy.array(sequence_pointers, dtype=numpy.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = numpy.array(document_indices, dtype=numpy.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = numpy.array(sequence_modes, dtype=numpy.int8)
            self.idx_writer.write(sequence_modes.tobytes(order='C'))
            del sequence_modes

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


class _IndexReader(object):
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    """

    def __init__(self, idx_path: str, multimodal: bool) -> None:

        log_single_rank(logger, logging.INFO, f"Load the {type(self).__name__} from {idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        log_single_rank(logger, logging.INFO, f"\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(logger, logging.INFO, f"\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(logger, logging.INFO, f"\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        self.sequence_modes = None
        if multimodal:
            log_single_rank(logger, logging.INFO, f"\tExtract the sequence modes")
            t_beg = time.time()
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        log_single_rank(logger, logging.INFO, f"> total number of sequences: {len(self)}")
        log_single_rank(
            logger,
            logging.INFO,
            f"> total number of documents: {self.document_indices.shape[0] - 1}",
        )

    def __del__(self) -> None:
        """Clean up the object"""
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class IndexedDatasetBuilder(object):
    """Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
        self, bin_path: str, dtype: Type[numpy.number] = numpy.int32
    ) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype

        self.sequence_lengths = []
        self.document_indices = [0]

    def add_document(
        self, input_ids: List[int], token_length: int, modes: Optional[List[int]] = None
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add

            lengths (List[int]): The lengths of each item in the document

            modes (Optional[List[int]], optional): The modes for each item in the document. Defaults to None.
        """
        np_array = numpy.array(input_ids, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend([token_length])
        self.document_indices.append(len(self.sequence_lengths))

    def add_index(self, path_prefix: str) -> None:
        """Add an entire IndexedDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        # Concatenate index
        index = _IndexReader(get_idx_path(path_prefix), multimodal=False)
        assert index.dtype == self.dtype

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)
        self.document_indices.extend((offset + index.document_indices)[1:])

        # Concatenate data
        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, None, self.document_indices)


def process_partition(kwargs):

    partition: str = kwargs['partition']
    output_prefix: str = kwargs['output_prefix']
    json_keys: List[str] = kwargs['json_keys']

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    level = 'document'
    for key in json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                        key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                        key, level)
        builders[key] = IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=DType.optimal_dtype(VOCAB_SIZE),
        )

    encoded_docs = datasets.load_dataset(partition, split='train', streaming=True)
    for json_dict in encoded_docs:
        for key in json_keys:
            content = json_dict[key]
            builders[key].add_document(content, len(content))
    for key in json_keys:
        builders[key].finalize(output_idx_files[key])


def convert_hf_dataset(dataset_path: str, json_keys: List[str], output_prefix: str, num_workers: int = 16):

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    level = 'document'

    dataset_names = sorted(os.listdir(dataset_path))
    dataset_names = [d for d in dataset_names if not d.startswith('.')]
    in_ss_out_names = [{'partition': os.path.join(dataset_path, d), 'output_prefix': d, 'json_keys': json_keys} for d in dataset_names]

    # process the dataset in parallel
    with ProcessPoolExecutor(num_workers) as executor:
        p = executor.map(process_partition, in_ss_out_names)
        for _ in tqdm(p, total=len(in_ss_out_names), desc="Processing dataset"):
            pass

    # collect different subsets into the same bin file
    for key in json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                      key, level)
        builders[key] = IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=DType.optimal_dtype(VOCAB_SIZE),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                            key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    convert_hf_dataset('/data/hf_dataset/myl_new_no_math/17_20241108_004951', ['input_ids'], '17_20241108_004951')

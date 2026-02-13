.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

MemMap Datasets
====================================================

MemMapDataset Index Generator
------------------------------------------------------------------------------

The :python:`MemMapDataset` requires an index file providing the necessary pointers into the raw data file. The :python:`MemMapDataset` can create the index file lazily, however, it is advised to create it beforehand. This can be done by running

.. code-block:: bash

  modalities data create_raw_index <path/to/jsonl/file>

The index will be created in the same directory as the raw data file. For further options you may look into the usage documentation via :bash:`modalities data create_raw_index --help`.

Packed Dataset Generator
--------------------------------------------------------------------------------

The :python:`PackedMemMapDatasetContinuous` and :python:`PackedMemMapDatasetMegatron` require a packed data file. To create the data file, you first have to generate a :python:`MemMapDataset` index file as described `above <memMapDataset-index-generator>`_. Assuming the index and raw data are located in the same directory, you can simply execute the following command:

.. code-block:: bash

  modalities data pack_encoded_data <path/to/jsonl/file>

The packed data file will be created in the same directory as the raw data file. For further options you may look into the usage documentation via :bash:`modalities data pack_encoded_data --help`.

Packed Data Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The packed data file is a bytestream containing both the tokenized data as well as an index denoting the start and length of the tokenized documents inside the bytestream. The data file consists of 3 concatenated parts:

header segment | data segment | index segment

* **header segment**: This section is a 8 bytes sized integer which encodes the length of the data segment in bytes.
* **data segment**: This section contains a concatenation of all documents in form of 4 bytes sized tokens. An end-of-sequence token is placed between consecutive documents.
* **index segment**: This section contains a pickled index which locates the documents inside the data segment. The index is basically a list of tuples, where each tuple contains the start position and length in bytes for the corresponding document, e.g., :python:`[(start_doc1, len_doc1), (start_doc2, len_doc2), ....]`.
